"""Multi-feature KLT + rigid affine stabilization.

Tracks N features across all frames with Lucas-Kanade optical flow, fits a
2D rigid transform (rotation + translation) per frame mapping current
positions back to frame-0 positions, then warpAffines each frame with that
inverse so the features hold steady. Output is HEVC via hevc_nvenc on CUDA
machines, hevc_videotoolbox on Apple Silicon.

Optional DLC H5 input lets you mask out the subject (bird/animal) so the
tracked features sit on the static background instead of the moving subject.

Usage:
  python klt_affine.py --input clip.mp4 --output stab.mp4 [--dlc_h5 clip.h5]
                       [--n_features 15] [--crop_w 0 --crop_h 0]
                       [--offset_x 0 --offset_y 0] [--bitrate 50M]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def pick_encoder(preview: bool = False) -> list[str]:
    """Prefer GPU encode; fall back to CPU x265. With preview=True, pick
    the fastest available options (lower quality is fine)."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        sys.exit("ffmpeg not found in PATH")
    if "hevc_nvenc" in out:
        return ["-c:v", "hevc_nvenc", "-preset", "p1" if preview else "p4"]
    if "hevc_videotoolbox" in out:
        # videotoolbox always runs at full speed; quality knob is bitrate.
        return ["-c:v", "hevc_videotoolbox"]
    return ["-c:v", "libx265", "-preset", "ultrafast" if preview else "fast"]


def bird_mask(frame_shape, h5_path: Path | None, skip: int = 0):
    """Build a binary mask that EXCLUDES the bird region (so we pick features
    on the background only). Returns (mask, bird_centroid) or (None, None).
    """
    if not h5_path:
        return None, None
    import pandas as pd
    H, W = frame_shape[:2]
    df = pd.read_hdf(h5_path)
    scorer = df.columns.get_level_values(0).unique()[0]
    a = df[scorer]["animal0"]
    parts = list(a.columns.get_level_values(0).unique())
    pts = []
    for p in parts:
        x = a[p]["x"].iloc[skip]; y = a[p]["y"].iloc[skip]; l = a[p]["likelihood"].iloc[skip]
        if l >= 0.3 and 0 <= x < W and 0 <= y < H:
            pts.append((x, y))
    if not pts:
        return None, None
    pts = np.array(pts)
    c = pts.mean(axis=0)
    r = max(np.linalg.norm(pts - c, axis=1)) * 1.5
    m = np.full((H, W), 255, dtype=np.uint8)
    cv2.circle(m, (int(c[0]), int(c[1])), int(r), 0, -1)
    return m, c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dlc_h5", default=None)
    ap.add_argument("--n_features", type=int, default=40)
    ap.add_argument("--crop_w", type=int, default=0, help="0 = source width")
    ap.add_argument("--crop_h", type=int, default=0, help="0 = source height")
    ap.add_argument("--offset_x", type=int, default=0)
    ap.add_argument("--offset_y", type=int, default=0)
    ap.add_argument("--bitrate", default="10M",
                    help="Output bitrate. Default 10M is good for previews; "
                         "bump to 30-50M for final exports.")
    ap.add_argument("--preview", action="store_true",
                    help="Speed-first encoding for previews: lower bitrate, fast preset, "
                         "1080p output. Overrides --bitrate.")
    ap.add_argument("--err_thresh", type=float, default=80.0,
                    help="Soft threshold — features above this for one frame are skipped that frame "
                         "but stay in the pool. Hard loss requires KLT status==0.")
    ap.add_argument("--ransac_thresh", type=float, default=3.0,
                    help="RANSAC reprojection threshold (pixels) for affine fit.")
    ap.add_argument("--feature_bbox", default=None,
                    help="x1,y1,x2,y2 — restrict feature selection to this rectangle. "
                         "Use to track only one static region (e.g. the perch branch).")
    ap.add_argument("--no_rotation", action="store_true",
                    help="Translation-only fit; skip rotation warp (cleaner edges when "
                         "the camera doesn't rotate much).")
    ap.add_argument("--initial_points", default=None,
                    help="Semicolon-separated list of seed feature coords in source "
                         "resolution: 'x1,y1;x2,y2;...'. Skips goodFeaturesToTrack — "
                         "use these exact points as the KLT starting set.")
    ap.add_argument("--tracks_json", default=None,
                    help="Use pre-computed tracks (e.g. from cotracker_track.py) "
                         "instead of running KLT. Expects JSON with per-frame [x,y,vis] "
                         "lists per point.")
    ap.add_argument("--vis_thresh", type=float, default=0.7,
                    help="Visibility threshold for using a point in the RANSAC fit "
                         "when tracks_json is provided.")
    ap.add_argument("--auto_pick", action="store_true",
                    help="Two-pass mode: pick many candidate features, track them through "
                         "the whole clip, keep the most durable subset for the real run. "
                         "Also reports frame ranges where feature count crashes (suggests "
                         "where to trim).")
    ap.add_argument("--auto_pick_pool", type=int, default=200,
                    help="Number of candidate features for --auto_pick first pass.")
    ap.add_argument("--consensus_filter", action="store_true", default=True,
                    help="(tracks_json mode) Drop tracks that disagree with the rigid-body "
                         "motion via per-frame RANSAC voting. Default ON. Catches features "
                         "that ended up on the bird / a moving object.")
    ap.add_argument("--no_consensus_filter", dest="consensus_filter", action="store_false")
    ap.add_argument("--consensus_min_rate", type=float, default=0.5,
                    help="Minimum per-track RANSAC-inlier rate to keep the track.")
    ap.add_argument("--debug_overlay", action="store_true",
                    help="(tracks_json mode) Write the SOURCE video with each tracked "
                         "point drawn as a numbered circle. Don't stabilise — just show "
                         "what is being tracked.")
    ap.add_argument("--border", choices=["replicate", "constant", "shrink"], default="constant",
                    help="What to do when the stabilized crop goes past source edges. "
                         "shrink: pre-scan all frames and pick a crop size that never reveals "
                         "borders (cleanest). constant: black borders. replicate: stretch "
                         "outermost pixel (causes smearing).")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    ok, frame0 = cap.read()
    if not ok:
        sys.exit("cannot read frame 0")
    H, W = frame0.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    crop_w = args.crop_w or W
    crop_h = args.crop_h or H

    mask, bird_c = bird_mask(frame0.shape, Path(args.dlc_h5) if args.dlc_h5 else None)
    if args.feature_bbox:
        x1, y1, x2, y2 = (int(v) for v in args.feature_bbox.split(","))
        bbox_mask = np.zeros((H, W), dtype=np.uint8)
        bbox_mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 255
        mask = bbox_mask if mask is None else cv2.bitwise_and(mask, bbox_mask)
        print(f"feature search restricted to bbox ({x1},{y1})-({x2},{y2})", flush=True)
    ref_center_x = (W / 2) + args.offset_x
    ref_center_y = (H / 2) + args.offset_y
    if bird_c is not None:
        ref_center_x = float(bird_c[0]) + args.offset_x
        ref_center_y = float(bird_c[1]) + args.offset_y

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    if args.tracks_json:
        # Pre-computed tracks (e.g. CoTracker3). Skip KLT, run the warp loop
        # directly off the JSON.
        import json
        with open(args.tracks_json) as f:
            td = json.load(f)
        tracks = np.array(td["tracks"])  # [N, P, 3] = (x, y, visibility)
        N = tracks.shape[0]
        P = tracks.shape[1]
        # Use frame 0 visible points as the "init" set
        init_full = tracks[0, :, :2].astype(np.float32)
        print(f"loaded {P} tracks over {N} frames; vis_thresh={args.vis_thresh}", flush=True)

        # Consensus filter: per-frame RANSAC inlier rate per track. Tracks that
        # disagree with the rigid-body motion (the bird flying away, a leaf
        # moving in wind) get rejected before the warp pass.
        if args.consensus_filter:
            inlier_count = np.zeros(P, dtype=np.int32)
            visible_count = np.zeros(P, dtype=np.int32)
            for fi in range(1, N):
                vis = tracks[fi, :, 2] >= args.vis_thresh
                visible_count += vis.astype(np.int32)
                if vis.sum() < 4:
                    continue
                cur = tracks[fi, vis, :2].astype(np.float32)
                src = init_full[vis]
                _, inl = cv2.estimateAffinePartial2D(
                    cur, src, method=cv2.RANSAC,
                    ransacReprojThreshold=args.ransac_thresh,
                    maxIters=500, confidence=0.99)
                if inl is None:
                    continue
                inl = inl.flatten().astype(bool)
                idx = np.where(vis)[0]
                inlier_count[idx[inl]] += 1
            rate = inlier_count / np.maximum(visible_count, 1)
            keep = rate >= args.consensus_min_rate
            if keep.sum() >= 4:
                dropped = (~keep).sum()
                print(f"consensus_filter: kept {keep.sum()}/{P} (median inlier rate {np.median(rate[keep]):.2f}); "
                      f"dropped {dropped} outliers (median outlier rate {np.median(rate[~keep]) if dropped else 0:.2f})", flush=True)
                tracks = tracks[:, keep, :]
                init_full = init_full[keep]
                P = int(keep.sum())
            else:
                print(f"consensus_filter: only {keep.sum()} survive, falling back to no filter", flush=True)

        if args.debug_overlay:
            # Write source-sized video with numbered circles at tracked positions.
            # No stabilisation, just visual debugging.
            enc = pick_encoder(preview=True)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
                   "-i", args.input, "-map", "0:v", "-map", "1:a?",
                   *enc, "-b:v", args.bitrate, "-tag:v", "hvc1",
                   "-pix_fmt", "yuv420p",
                   "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                   "-bsf:v", "hevc_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
                   "-c:a", "aac", "-b:a", "192k", "-shortest", args.output]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            assert proc.stdin
            # Distinct colors per index
            rng = np.random.default_rng(0)
            colors = (rng.uniform(60, 255, size=(P, 3))).astype(int).tolist()
            # First frame
            f0 = frame0.copy()
            for pi in range(P):
                vis = tracks[0, pi, 2] >= args.vis_thresh
                x, y = int(tracks[0, pi, 0]), int(tracks[0, pi, 1])
                col = colors[pi] if vis else (60, 60, 60)
                cv2.circle(f0, (x, y), 18, col, 3)
                cv2.putText(f0, str(pi), (x + 22, y + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            proc.stdin.write(f0.tobytes())
            for fi in range(1, N):
                ok, frame = cap.read()
                if not ok: break
                for pi in range(P):
                    vis = tracks[fi, pi, 2] >= args.vis_thresh
                    x, y = int(tracks[fi, pi, 0]), int(tracks[fi, pi, 1])
                    col = colors[pi] if vis else (60, 60, 60)
                    cv2.circle(frame, (x, y), 18, col, 3)
                    cv2.putText(frame, str(pi), (x + 22, y + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
                proc.stdin.write(frame.tobytes())
                if fi % 100 == 0:
                    print(f"  {fi}/{N} (debug overlay)", flush=True)
            cap.release()
            proc.stdin.close()
            proc.wait()
            print(f"debug overlay -> {args.output}", flush=True)
            return

        # Pre-scan: compute per-frame median displacement so we can size the crop
        # to never reveal borders (when --border=shrink).
        if args.border == "shrink":
            shifts = []
            for fi in range(N):
                vis = tracks[fi, :, 2] >= args.vis_thresh
                if vis.sum() < 3:
                    continue
                disp = tracks[fi, vis, :2] - init_full[vis]
                shifts.append((float(np.median(disp[:, 0])), float(np.median(disp[:, 1]))))
            if shifts:
                arr = np.array(shifts)
                max_dx = int(np.ceil(np.abs(arr[:, 0]).max()))
                max_dy = int(np.ceil(np.abs(arr[:, 1]).max()))
                # Shrink crop by 2*max so the warped frame always fully covers it.
                new_w = max(64, crop_w - 2 * max_dx)
                new_h = max(64, crop_h - 2 * max_dy)
                if (new_w, new_h) != (crop_w, crop_h):
                    print(f"shrink: max shift ({max_dx}, {max_dy}) -> crop {crop_w}x{crop_h} -> {new_w}x{new_h}", flush=True)
                    crop_w, crop_h = new_w, new_h
        border_mode = {
            "replicate": cv2.BORDER_REPLICATE,
            "constant": cv2.BORDER_CONSTANT,
            "shrink": cv2.BORDER_CONSTANT,  # shouldn't matter after shrink
        }[args.border]
        # Output pipeline
        enc = pick_encoder(preview=args.preview)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{crop_w}x{crop_h}", "-r", f"{fps}",
               "-i", "-", "-i", args.input,
               "-map", "0:v", "-map", "1:a?",
               *enc, "-b:v", args.bitrate, "-tag:v", "hvc1",
               "-pix_fmt", "yuv420p",
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
               "-bsf:v", "hevc_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
               "-c:a", "aac", "-b:a", "192k", "-shortest", args.output]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin
        x1 = int(ref_center_x - crop_w / 2)
        y1 = int(ref_center_y - crop_h / 2)
        proc.stdin.write(frame0[y1:y1 + crop_h, x1:x1 + crop_w].tobytes())
        T_out = np.array([[1, 0, crop_w / 2 - ref_center_x],
                          [0, 1, crop_h / 2 - ref_center_y]], dtype=np.float32)
        out_M = T_out.copy()
        for fi in range(1, N):
            ok, frame = cap.read()
            if not ok: break
            visible = tracks[fi, :, 2] >= args.vis_thresh
            cur = tracks[fi, :, :2].astype(np.float32)
            if visible.sum() < 3:
                stab = cv2.warpAffine(frame, out_M, (crop_w, crop_h),
                                      flags=cv2.INTER_CUBIC, borderMode=border_mode)
                proc.stdin.write(stab.tobytes())
                if fi % 100 == 0:
                    print(f"  {fi}/{N} ({int(visible.sum())} visible — holding last M)", flush=True)
                continue
            if args.no_rotation:
                disp = cur[visible] - init_full[visible]
                tx, ty = float(np.median(disp[:, 0])), float(np.median(disp[:, 1]))
                M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
            else:
                M, _ = cv2.estimateAffinePartial2D(
                    cur[visible], init_full[visible],
                    method=cv2.RANSAC, ransacReprojThreshold=args.ransac_thresh,
                    maxIters=2000, confidence=0.999)
                if M is None:
                    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            M3 = np.vstack([M, [0, 0, 1]])
            T3 = np.vstack([T_out, [0, 0, 1]])
            out_M = (T3 @ M3)[:2]
            stab = cv2.warpAffine(frame, out_M, (crop_w, crop_h),
                                  flags=cv2.INTER_CUBIC, borderMode=border_mode)
            proc.stdin.write(stab.tobytes())
            if fi % 100 == 0:
                print(f"  {fi}/{N} ({int(visible.sum())}/{P} visible)", flush=True)
        cap.release()
        proc.stdin.close()
        proc.wait()
        print(f"done -> {args.output}", flush=True)
        return

    if args.auto_pick:
        # Pass 1: track a big pool of candidate features through the whole clip,
        # then keep the most durable for the real stab pass.
        cand = cv2.goodFeaturesToTrack(
            gray0, maxCorners=args.auto_pick_pool, qualityLevel=0.005,
            minDistance=20, mask=mask, blockSize=11)
        if cand is None or len(cand) < 5:
            sys.exit(f"auto_pick: too few candidates ({0 if cand is None else len(cand)})")
        npc = len(cand)
        print(f"auto_pick pass-1: tracking {npc} candidates through {N} frames", flush=True)
        survival = np.zeros(npc, dtype=np.int32)
        alive = np.ones(npc, dtype=bool)
        per_frame_alive = []
        prev_gray_p1 = gray0
        prev_pts_p1 = cand.copy()
        lk_p1 = dict(winSize=(31, 31), maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        for fi in range(1, N):
            ok, frame = cap.read()
            if not ok: break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            nxt, status, err = cv2.calcOpticalFlowPyrLK(prev_gray_p1, g, prev_pts_p1, None, **lk_p1)
            alive &= status.flatten() == 1
            # Also reject features that wandered way off (rough sanity)
            disp = nxt.reshape(-1, 2) - cand.reshape(-1, 2)
            ddist = np.linalg.norm(disp, axis=1)
            alive &= ddist < 500
            survival += alive.astype(np.int32)
            per_frame_alive.append(int(alive.sum()))
            prev_gray_p1 = g
            prev_pts_p1 = nxt
        # Trim suggestions: find frames where alive count drops sharply
        if per_frame_alive:
            arr = np.array(per_frame_alive)
            start = arr[0] if arr[0] else npc
            crash = np.where(arr < max(5, start * 0.5))[0]
            if len(crash):
                first = int(crash[0])
                print(f"TRIM_SUGGEST: feature count crashes at frame {first+1} (~{(first+1)/fps:.1f}s); "
                      f"consider --duration_sec {(first+1)/fps:.1f}", flush=True)
            else:
                print(f"TRIM_SUGGEST: clip is stable end-to-end (no crash detected)", flush=True)
        # Pick top by survival; if user wanted N features, take top N.
        top_n = min(args.n_features, npc)
        order = np.argsort(survival)[::-1][:top_n]
        print(f"auto_pick pass-2: keeping top {top_n}/{npc} by survival "
              f"(median survival of kept = {int(np.median(survival[order]))}/{N})", flush=True)
        pick = cand[order].astype(np.float32)
        # Rewind for pass 2
        cap.release()
        cap = cv2.VideoCapture(args.input)
        ok, frame0 = cap.read()
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        flat = pick.reshape(-1, 2)
        order = np.arange(len(flat))
        n_pick = len(flat)
    elif args.initial_points:
        pts = []
        for chunk in args.initial_points.split(";"):
            x, y = chunk.split(",")
            pts.append([float(x), float(y)])
        flat = np.array(pts, dtype=np.float32)
        order = np.arange(len(flat))
        n_pick = len(flat)
        print(f"using {n_pick} caller-provided initial points", flush=True)
    else:
        corners = cv2.goodFeaturesToTrack(
            gray0, maxCorners=400, qualityLevel=0.005,
            minDistance=30, mask=mask, blockSize=11)
        if corners is None or len(corners) < 3:
            sys.exit(f"too few features ({0 if corners is None else len(corners)}) — need ≥3")
        flat = corners.reshape(-1, 2)
        if bird_c is not None:
            order = np.argsort(np.linalg.norm(flat - bird_c, axis=1))
        else:
            order = np.arange(len(flat))  # strongest-first from goodFeaturesToTrack
        n_pick = min(args.n_features, len(flat))
        if n_pick < args.n_features:
            print(f"warn: only {len(flat)} features in masked region; using {n_pick} (requested {args.n_features})", flush=True)
    pick = flat[order[:n_pick]].astype(np.float32).reshape(-1, 1, 2)
    init = pick.reshape(-1, 2).copy()
    print(f"{n_pick} features locked, ref_center=({ref_center_x:.0f},{ref_center_y:.0f}) crop {crop_w}x{crop_h}", flush=True)

    enc = pick_encoder(preview=args.preview)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{crop_w}x{crop_h}", "-r", f"{fps}",
           "-i", "-",
           "-i", args.input,
           "-map", "0:v", "-map", "1:a?",
           *enc, "-b:v", args.bitrate, "-tag:v", "hvc1",
           "-pix_fmt", "yuv420p",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-bsf:v", "hevc_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
           "-c:a", "aac", "-b:a", "192k", "-shortest",
           args.output]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin

    lk = dict(winSize=(31, 31), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    prev_gray = gray0
    prev_pts = pick.copy()
    active = np.ones(n_pick, dtype=bool)
    # Border mode for the KLT branch (the tracks_json branch sets its own above).
    border_mode = {
        "replicate": cv2.BORDER_REPLICATE,
        "constant": cv2.BORDER_CONSTANT,
        "shrink": cv2.BORDER_CONSTANT,
    }[args.border]

    x1 = int(ref_center_x - crop_w / 2)
    y1 = int(ref_center_y - crop_h / 2)
    proc.stdin.write(frame0[y1:y1 + crop_h, x1:x1 + crop_w].tobytes())

    for fi in range(1, N):
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk)
        status = status.flatten(); err = err.flatten()
        # Hard loss only when KLT reports status==0 (feature went off-screen
        # or couldn't be tracked at all). Soft transient errors stay in the
        # pool so RANSAC can pick them back up next frame.
        active &= status == 1
        cur = next_pts.reshape(-1, 2)
        # Per-frame "fit-eligible" subset: alive AND not currently noisy.
        eligible = active & (err <= args.err_thresh)
        if eligible.sum() < 3:
            # Not enough clean signal this frame — hold the last good transform.
            prev_gray = gray
            prev_pts = next_pts
            stab = cv2.warpAffine(frame, out_M, (crop_w, crop_h),
                                  flags=cv2.INTER_CUBIC, borderMode=border_mode)
            proc.stdin.write(stab.tobytes())
            if fi % 100 == 0:
                print(f"  {fi}/{N} ({eligible.sum()}/{active.sum()} eligible — holding last M)", flush=True)
            continue
        if args.no_rotation:
            # Translation-only fit: median displacement vector.
            disp = cur[eligible] - init[eligible]
            tx, ty = float(np.median(disp[:, 0])), float(np.median(disp[:, 1]))
            M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        else:
            # Full RANSAC affine — handles per-frame outliers (e.g., a leaf
            # that briefly swayed) without dropping them from the pool forever.
            M, _ = cv2.estimateAffinePartial2D(
                cur[eligible], init[eligible],
                method=cv2.RANSAC, ransacReprojThreshold=args.ransac_thresh,
                maxIters=2000, confidence=0.999)
            if M is None:
                M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        T_out = np.array([[1, 0, crop_w / 2 - ref_center_x],
                          [0, 1, crop_h / 2 - ref_center_y]], dtype=np.float32)
        M3 = np.vstack([M, [0, 0, 1]])
        T3 = np.vstack([T_out, [0, 0, 1]])
        out_M = (T3 @ M3)[:2]
        stab = cv2.warpAffine(frame, out_M, (crop_w, crop_h),
                              flags=cv2.INTER_CUBIC, borderMode=border_mode)
        proc.stdin.write(stab.tobytes())
        prev_gray = gray
        prev_pts = next_pts
        if fi % 100 == 0:
            print(f"  {fi}/{N} ({active.sum()}/{n_pick} active)", flush=True)

    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"done -> {args.output}")


if __name__ == "__main__":
    main()
