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
    ap.add_argument("--warps_json", default=None,
                    help="Use pre-computed per-frame warps (from klt_track.py's "
                         "frame-to-frame accumulator). JSON {warps:[[2x3],...]} mapping "
                         "frame_i->frame_0. Skips fitting; goes straight to crop/refine.")
    ap.add_argument("--vis_thresh", type=float, default=0.7,
                    help="Visibility threshold for using a point in the RANSAC fit "
                         "when tracks_json is provided.")
    ap.add_argument("--deglitch", type=float, default=0,
                    help="(tracks/warps) Flag frames whose warp translation deviates more "
                         "than this many px from its median-filtered trajectory (occlusion "
                         "spikes) and replace them by interpolation. 0=off; try 40.")
    ap.add_argument("--deglitch_win", type=int, default=21,
                    help="Median-filter window for --deglitch.")
    ap.add_argument("--warp_smooth", type=int, default=0,
                    help="(tracks_json) Moving-average window over the per-frame warp to "
                         "remove residual estimation jitter. Safe for a static lock (no lag). "
                         "0/1 = off; try 3-5.")
    ap.add_argument("--refine_bbox", default=None,
                    help="(tracks_json) x1,y1,x2,y2 source px — after the coarse track-based "
                         "warp, ECC-refine each frame against FRAME 0 on this textured patch "
                         "for a sub-pixel lock (KLT-grade precision with tracker robustness). "
                         "The coarse warp puts the residual within ~2px, so ECC can't wander.")
    ap.add_argument("--anchor_bbox", default=None,
                    help="(tracks_json) x1,y1,x2,y2 source px — like --refine_bbox but uses "
                         "normalized cross-correlation (matchTemplate) instead of ECC. NCC picks "
                         "the GLOBAL correlation peak in a +/-pad search window, so repetitive "
                         "texture (lichen/bark) can't trap it in a wrong local minimum the way "
                         "ECC's gradient descent does. This is the same metric used to measure "
                         "lock, so it drives measured displacement to ~0 directly. Translational.")
    ap.add_argument("--anchor_pad", type=int, default=120,
                    help="Search half-range (px) for --anchor_bbox NCC matching.")
    ap.add_argument("--anchor_min_cc", type=float, default=0.5,
                    help="Min NCC peak to accept an --anchor_bbox correction.")
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
    ap.add_argument("--auto_crop", action="store_true",
                    help="(tracks_json) Size the crop to the largest border-safe rect at "
                         "--auto_crop_pct, maximizing usable area.")
    ap.add_argument("--auto_crop_pct", type=float, default=2.0,
                    help="Percentile slack for auto_crop: ignore the worst this-%% of frames "
                         "per edge (e.g. a brief takeoff lurch) when sizing the crop.")
    ap.add_argument("--aspect", default="16:9", help="Aspect for --auto_crop.")
    ap.add_argument("--crop_rect", default=None,
                    help="x0,y0,w,h source px — explicit crop rectangle (overrides --auto_crop). "
                         "Use a rotation-aware black-free rect computed offline; combine with "
                         "--out_w/--out_h to downscale (zoom out) to a standard output size.")
    ap.add_argument("--out_w", type=int, default=0, help="Output width (downscale crop_rect). 0=crop width.")
    ap.add_argument("--out_h", type=int, default=0, help="Output height. 0=crop height.")
    ap.add_argument("--lut", default=None,
                    help="Path to a .cube 3D LUT (e.g. SV2 Look S-Log3->Rec709). Applied in the "
                         "encode in a float pipeline, output 10-bit p010 with dither to protect "
                         "smooth-sky gradients from banding.")
    ap.add_argument("--read_bits", type=int, choices=[8, 16], default=8,
                    help="8 = decode via OpenCV (8-bit, fast). 16 = decode the source as 16-bit "
                         "(bgr48le) so the warp/downscale preserve 10-bit precision — no banding "
                         "in smooth gradients. Use 16 for graded 10-bit sources / final exports.")
    ap.add_argument("--x265", action="store_true",
                    help="Encode with libx265 (software, slow, high quality 10-bit) instead of the "
                         "HW encoder. Use for final exports; pairs with --read_bits 16.")
    ap.add_argument("--x265_crf", type=int, default=16,
                    help="CRF for --x265 (lower=better; 16 is near-transparent).")
    ap.add_argument("--bias_x", type=int, default=0,
                    help="Shift the auto_crop window horizontally within the safe region.")
    ap.add_argument("--bias_y", type=int, default=0,
                    help="Shift the auto_crop window vertically (negative = up, toward bird).")
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

    if args.tracks_json or args.warps_json:
        # Pre-computed motion. Either per-point tracks (CoTracker / local KLT, we
        # fit the per-frame warp here) or per-frame warps directly (--warps_json
        # from klt_track.py's frame-to-frame accumulator).
        import json
        if args.warps_json:
            with open(args.warps_json) as f:
                wd = json.load(f)
            Ms = [np.asarray(w, dtype=np.float32) for w in wd["warps"]]
            N = len(Ms)
            print(f"loaded {N} precomputed warps from {args.warps_json}", flush=True)
        else:
            with open(args.tracks_json) as f:
                td = json.load(f)
            tracks = np.array(td["tracks"])  # [N, P, 3] = (x, y, visibility)
            N = tracks.shape[0]
            P = tracks.shape[1]
            init_full = tracks[0, :, :2].astype(np.float32)
            print(f"loaded {P} tracks over {N} frames; vis_thresh={args.vis_thresh}", flush=True)

        # Consensus filter: per-frame RANSAC inlier rate per track. Tracks that
        # disagree with the rigid-body motion (the bird flying away, a leaf
        # moving in wind) get rejected before the warp pass.
        if args.tracks_json and args.consensus_filter:
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

        # ---- Pass A: compute the stabilizing warp M (current->frame0) for every
        # frame from the tracks alone (no video read). M is applied to a frame to
        # bring its tracked points back to their frame-0 positions. ----
        def similarity_fit(src, dst):
            """Closed-form similarity (rotation + uniform scale + translation)
            mapping src->dst, using ALL points (deterministic — no RANSAC jitter).
            Umeyama 1991."""
            src_m = src.mean(0); dst_m = dst.mean(0)
            sc = src - src_m; dc = dst - dst_m
            Hcov = (sc.T @ dc) / len(src)
            U, S, Vt = np.linalg.svd(Hcov)
            d = 1.0 if np.linalg.det(Vt.T @ U.T) > 0 else -1.0
            D = np.diag([1.0, d])
            R = (Vt.T @ D @ U.T)
            var = (sc ** 2).sum() / len(src)
            s = (S @ np.array([1.0, d])) / var if var > 1e-9 else 1.0
            t = dst_m - s * (R @ src_m)
            M = np.zeros((2, 3), np.float32)
            M[:2, :2] = (s * R).astype(np.float32)
            M[:, 2] = t.astype(np.float32)
            return M

        def robust_fit(src, dst):
            """similarity_fit + one IRLS reject pass (drop points >3x median
            residual, refit) — deterministic robustness without RANSAC."""
            M = similarity_fit(src, dst)
            proj = (M[:, :2] @ src.T).T + M[:, 2]
            res = np.linalg.norm(proj - dst, axis=1)
            med = np.median(res)
            keep = res <= max(3.0 * med, 1.0)
            if 3 <= keep.sum() < len(src):
                M = similarity_fit(src[keep], dst[keep])
            return M

        Ms = Ms if args.warps_json else []
        good_fit = np.zeros(N, dtype=bool)
        ident = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        last_M = ident.copy()
        for fi in range(N if args.tracks_json else 0):
            if fi == 0:
                Ms.append(ident.copy()); good_fit[0] = True; continue
            visible = tracks[fi, :, 2] >= args.vis_thresh
            cur = tracks[fi, :, :2].astype(np.float32)
            if visible.sum() < 3:
                Ms.append(last_M); continue
            if args.no_rotation:
                disp = init_full[visible] - cur[visible]   # cur -> frame0
                M = np.array([[1, 0, float(np.median(disp[:, 0]))],
                              [0, 1, float(np.median(disp[:, 1]))]], dtype=np.float32)
            else:
                M = robust_fit(cur[visible], init_full[visible])
            M = M.astype(np.float32)
            Ms.append(M); last_M = M
            good_fit[fi] = True

        # Bridge visibility-collapse gaps by interpolating the warp between the
        # bracketing good frames (camera motion is continuous; gaps are short).
        # Holding a frozen warp through a gap lets the scene drift — this doesn't.
        n_interp = 0
        gi = np.where(good_fit)[0] if args.tracks_json else np.array([], dtype=int)
        for a, b in zip(gi[:-1], gi[1:]):
            if b - a <= 1:
                continue
            Ma, Mb = Ms[a], Ms[b]
            angA = np.arctan2(Ma[1, 0], Ma[0, 0]); angB = np.arctan2(Mb[1, 0], Mb[0, 0])
            scA = np.hypot(Ma[0, 0], Ma[1, 0]);    scB = np.hypot(Mb[0, 0], Mb[1, 0])
            for fi in range(a + 1, b):
                t = (fi - a) / (b - a)
                ang = angA + t * (angB - angA)
                sc = scA + t * (scB - scA)
                tx = Ma[0, 2] + t * (Mb[0, 2] - Ma[0, 2])
                ty = Ma[1, 2] + t * (Mb[1, 2] - Ma[1, 2])
                Ms[fi] = np.array([[sc * np.cos(ang), -sc * np.sin(ang), tx],
                                   [sc * np.sin(ang),  sc * np.cos(ang), ty]], dtype=np.float32)
                n_interp += 1
        if args.tracks_json:
            print(f"fit: {int(good_fit.sum())}/{N} frames direct, {n_interp} interpolated, "
                  f"{N - int(good_fit.sum()) - n_interp} held", flush=True)

        # Deglitch: detect frames where the warp spikes away from its robust
        # (median-filtered) trajectory — these are occlusion captures (the wing
        # dragging the fit). Real camera motion stays near the median; a wing
        # spike (100px+) is far. Replace flagged frames by linear interpolation
        # between the nearest clean frames.
        if args.deglitch > 0:
            arr = np.stack(Ms).astype(np.float64)          # [N,2,3]
            txy = arr[:, :, 2]                              # [N,2]
            k = args.deglitch_win | 1                       # odd window
            pad = k // 2
            padded = np.pad(txy, ((pad, pad), (0, 0)), mode="edge")
            medt = np.stack([np.median(padded[i:i + k], axis=0) for i in range(N)])
            dev = np.linalg.norm(txy - medt, axis=1)
            bad = dev > args.deglitch
            # also flag a few frames around each spike (capture ramps in/out)
            badi = np.where(bad)[0]
            for b in badi:
                bad[max(0, b - 2):min(N, b + 3)] = True
            goodi = np.where(~bad)[0]
            n_fix = 0
            if len(goodi) >= 2:
                for a, c in zip(goodi[:-1], goodi[1:]):
                    if c - a <= 1:
                        continue
                    Ma, Mc = Ms[a], Ms[c]
                    angA = np.arctan2(Ma[1, 0], Ma[0, 0]); angC = np.arctan2(Mc[1, 0], Mc[0, 0])
                    for fi in range(a + 1, c):
                        t = (fi - a) / (c - a)
                        ang = angA + t * (angC - angA)
                        tx = Ma[0, 2] + t * (Mc[0, 2] - Ma[0, 2])
                        ty = Ma[1, 2] + t * (Mc[1, 2] - Ma[1, 2])
                        Ms[fi] = np.array([[np.cos(ang), -np.sin(ang), tx],
                                           [np.sin(ang), np.cos(ang), ty]], dtype=np.float32)
                        n_fix += 1
            print(f"deglitch: {int(bad.sum())} frames flagged (dev>{args.deglitch}px), "
                  f"{n_fix} interpolated", flush=True)

        # Optional light temporal smoothing of the warp to remove residual
        # per-frame estimation jitter (safe for a static lock — no subject lag).
        if args.warp_smooth > 1:
            k = args.warp_smooth
            arr = np.stack(Ms).astype(np.float64)        # [N,2,3]
            pad = k // 2
            padded = np.pad(arr, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            kernel = np.ones(k) / k
            out = np.empty_like(arr)
            for i in range(2):
                for j in range(3):
                    out[:, i, j] = np.convolve(padded[:, i, j], kernel, mode="valid")[:N]
            Ms = [out[fi].astype(np.float32) for fi in range(N)]
            print(f"warp_smooth: moving-average window {k}", flush=True)

        # ---- Crop: border-safe region in stabilized (frame-0) space. A dst pixel
        # is valid iff it lies in M @ (frame rect); the inner AABB per frame, taken
        # at a percentile across frames, is the largest crop that stays border-free
        # for (100 - 2*pct)% of frames (ignoring the worst few, e.g. the takeoff). ----
        corners = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], dtype=np.float32).T
        Ls, Rs, Ts, Bs = [], [], [], []
        for M in Ms:
            q = (np.vstack([M, [0, 0, 1]]) @ corners)[:2].T   # 4x2 in dst space
            tl, tr, br, bl = q
            Ls.append(max(tl[0], bl[0])); Rs.append(min(tr[0], br[0]))
            Ts.append(max(tl[1], tr[1])); Bs.append(min(bl[1], br[1]))
        # For a crop edge to be border-safe on (100-pct)% of frames: the left/top
        # must sit at the *high* percentile of per-frame left/top edges, and the
        # right/bottom at the *low* percentile of right/bottom edges. (Getting these
        # sides backwards yields the full frame — i.e. no crop.)
        pct = args.auto_crop_pct
        safe_l = max(0, np.percentile(Ls, 100 - pct))
        safe_r = min(W, np.percentile(Rs, pct))
        safe_t = max(0, np.percentile(Ts, 100 - pct))
        safe_b = min(H, np.percentile(Bs, pct))
        print(f"safe region (pct={pct}): x[{safe_l:.0f},{safe_r:.0f}] y[{safe_t:.0f},{safe_b:.0f}] "
              f"= {safe_r - safe_l:.0f}x{safe_b - safe_t:.0f}", flush=True)

        if args.crop_rect:
            cx0, cy0, crop_w, crop_h = (int(v) for v in args.crop_rect.split(","))
        elif args.auto_crop:
            aw, ah = (int(v) for v in args.aspect.split(":"))
            avail_w, avail_h = safe_r - safe_l, safe_b - safe_t
            if avail_w / avail_h > aw / ah:
                crop_h = int(avail_h); crop_w = int(crop_h * aw / ah)
            else:
                crop_w = int(avail_w); crop_h = int(crop_w * ah / aw)
            crop_w -= crop_w % 2; crop_h -= crop_h % 2
            cx = (safe_l + safe_r) / 2 + args.bias_x
            cy = (safe_t + safe_b) / 2 + args.bias_y
            cx0 = int(round(max(safe_l, min(cx - crop_w / 2, safe_r - crop_w))))
            cy0 = int(round(max(safe_t, min(cy - crop_h / 2, safe_b - crop_h))))
        else:
            cx0 = int(ref_center_x - crop_w / 2)
            cy0 = int(ref_center_y - crop_h / 2)
        # Output size: downscale the crop rect to (out_w,out_h) if given — this is
        # how we "zoom out" (capture a larger source rect, fit a standard frame).
        out_w = args.out_w or crop_w; out_h = args.out_h or crop_h
        out_w -= out_w % 2; out_h -= out_h % 2
        sx = out_w / crop_w; sy = out_h / crop_h
        print(f"crop {crop_w}x{crop_h} at ({cx0},{cy0}) -> output {out_w}x{out_h} "
              f"(scale {sx:.4f})", flush=True)

        border_mode = cv2.BORDER_REPLICATE if args.border == "replicate" else cv2.BORDER_CONSTANT
        T_out = np.array([[sx, 0, -cx0 * sx], [0, sy, -cy0 * sy], [0, 0, 1]], dtype=np.float32)

        # ---- Pass B: render. ----
        # Optional per-frame sub-pixel ECC refinement against FRAME 0 on a small
        # textured patch. The coarse track-based warp puts the residual within a
        # couple of px, so ECC starts inside its convergence basin every frame —
        # none of the drift/ambiguity failure modes of standalone ECC.
        refine = None
        if args.refine_bbox:
            rx1, ry1, rx2, ry2 = (int(v) for v in args.refine_bbox.split(","))
            rpad = 120
            rwx1 = max(0, rx1 - rpad); rwy1 = max(0, ry1 - rpad)
            rwx2 = min(W, rx2 + rpad); rwy2 = min(H, ry2 + rpad)
            rww, rwh = rwx2 - rwx1, rwy2 - rwy1
            gray0_full = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            tmpl_win = gray0_full[rwy1:rwy2, rwx1:rwx2].copy()
            rmask = np.zeros((rwh, rww), np.uint8)
            cv2.rectangle(rmask, (rx1 - rwx1, ry1 - rwy1), (rx2 - rwx1, ry2 - rwy1), 255, -1)
            T_shift = np.array([[1, 0, -rwx1], [0, 1, -rwy1], [0, 0, 1]], dtype=np.float64)
            T_shift_inv = np.linalg.inv(T_shift)
            ecc_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-5)
            refine = dict(n_ok=0, n_skip=0)
            print(f"refine: patch ({rx1},{ry1})-({rx2},{ry2}) window {rww}x{rwh}", flush=True)

        # NCC template-anchor: re-anchor every frame's knot to its frame-0
        # position via matchTemplate. The coarse track warp brings the frame into
        # frame-0 coords; we then NCC the frame-0 knot template inside a padded
        # search window and translate the warp by the residual. Global-peak search
        # immune to the repetitive-bark local minima that defeat ECC.
        anchor = None
        if args.anchor_bbox:
            ax1, ay1, ax2, ay2 = (int(v) for v in args.anchor_bbox.split(","))
            apad = args.anchor_pad
            awx1 = max(0, ax1 - apad); awy1 = max(0, ay1 - apad)
            awx2 = min(W, ax2 + apad); awy2 = min(H, ay2 + apad)
            awW, awH = awx2 - awx1, awy2 - awy1
            g0a = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY).astype(np.float32)
            anchor_tmpl = g0a[ay1:ay2, ax1:ax2].copy()        # frame-0 knot
            # Expected peak top-left in the window frame if perfectly aligned:
            exp_x, exp_y = ax1 - awx1, ay1 - awy1
            T_shift_a = np.array([[1, 0, -awx1], [0, 1, -awy1], [0, 0, 1]], dtype=np.float64)
            anchor = dict(n_ok=0, n_skip=0)
            print(f"anchor(NCC): knot ({ax1},{ay1})-({ax2},{ay2}) window {awW}x{awH} "
                  f"search +/-{apad}px", flush=True)

        # 16-bit pipeline: decode the source as bgr48le so the warp + downscale
        # keep 10-bit precision (no banding in smooth sky); encode 10-bit out.
        read16 = args.read_bits == 16
        in_pix = "bgr48le" if read16 else "bgr24"
        # Grade in-pass: float pipeline -> lut3d. Output 10-bit when grading or 16-bit reading.
        ten_bit = read16 or bool(args.lut)
        if args.lut:
            lutp = args.lut.replace("\\", "/").replace(":", "\\:")
            vf = ["-vf", f"format=gbrpf32le,lut3d='{lutp}':interp=tetrahedral,format=p010le"]
            out_pix = "p010le"
        else:
            vf = []
            out_pix = "p010le" if ten_bit else "yuv420p"
        if args.x265:
            enc = ["-c:v", "libx265", "-preset", "slow",
                   "-crf", str(args.x265_crf), "-x265-params", "profile=main10:log-level=error"]
            rate = []
        else:
            enc = pick_encoder(preview=args.preview)
            rate = ["-b:v", args.bitrate]
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
               "-f", "rawvideo", "-pix_fmt", in_pix,
               "-s", f"{out_w}x{out_h}", "-r", f"{fps}",
               "-i", "-", "-i", args.input,
               "-map", "0:v", "-map", "1:a?", *vf,
               *enc, *rate, "-tag:v", "hvc1",
               "-pix_fmt", out_pix,
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
               "-bsf:v", "hevc_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
               "-c:a", "aac", "-b:a", "192k", "-shortest", args.output]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin
        cap.release()
        # Frame source: a raw ffmpeg decoder for 16-bit, else OpenCV.
        dec = None
        if read16:
            frame_bytes = W * H * 3 * 2
            dec = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", args.input,
                 "-f", "rawvideo", "-pix_fmt", "bgr48le", "-"],
                stdout=subprocess.PIPE, bufsize=frame_bytes)
        else:
            cap = cv2.VideoCapture(args.input)   # rewind: frame0 consumed at top of main()
        for fi in range(N):
            if read16:
                buf = dec.stdout.read(frame_bytes)
                if len(buf) < frame_bytes: break
                frame = np.frombuffer(buf, np.uint16).reshape(H, W, 3)
                gray8 = cv2.cvtColor((frame >> 8).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                ok, frame = cap.read()
                if not ok: break
                gray8 = None
            M3 = np.vstack([Ms[fi], [0, 0, 1]]).astype(np.float64)
            if refine is not None and fi > 0:
                # Coarse-align just the refine window with the track warp, then
                # ECC the residual against frame 0 — re-anchors absolutely, so any
                # coarse drift (incl. re-seed reference drift) is corrected, and
                # the lock is sub-pixel. Coarse-to-fine so it captures big (~60px)
                # residuals that a single full-res ECC pass would miss.
                win_M = (T_shift @ M3)[:2].astype(np.float32)
                if read16:
                    aligned_win = cv2.warpAffine(gray8, win_M, (rww, rwh), flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                else:
                    aligned_win = cv2.cvtColor(cv2.warpAffine(frame, win_M, (rww, rwh), flags=cv2.INTER_LINEAR),
                                               cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                wecc = np.eye(2, 3, dtype=np.float32)
                ok_ecc = False; cc = 0.0
                for sclvl in (0.25, 0.5, 1.0):
                    tl = tmpl_win if sclvl == 1.0 else cv2.resize(tmpl_win, None, fx=sclvl, fy=sclvl)
                    al = aligned_win if sclvl == 1.0 else cv2.resize(aligned_win, None, fx=sclvl, fy=sclvl)
                    ml = rmask if sclvl == 1.0 else cv2.resize(rmask, None, fx=sclvl, fy=sclvl, interpolation=cv2.INTER_NEAREST)
                    w = wecc.copy(); w[:, 2] *= sclvl
                    try:
                        cc, w = cv2.findTransformECC(tl, al, w, cv2.MOTION_EUCLIDEAN, ecc_crit, ml, 5)
                        w[:, 2] /= sclvl
                        wecc = w; ok_ecc = True
                    except cv2.error:
                        ok_ecc = False; break
                if ok_ecc and np.isfinite(wecc).all() and cc > 0.4:
                    R3 = T_shift_inv @ np.linalg.inv(np.vstack([wecc, [0, 0, 1]])) @ T_shift
                    M3 = R3 @ M3
                    refine["n_ok"] += 1
                else:
                    refine["n_skip"] += 1
            if anchor is not None and fi > 0:
                win_M = (T_shift_a @ M3)[:2].astype(np.float32)
                if read16:   # warp the 8-bit gray (consistent range with the template)
                    aligned = cv2.warpAffine(gray8, win_M, (awW, awH), flags=cv2.INTER_LINEAR).astype(np.float32)
                else:
                    aligned = cv2.cvtColor(cv2.warpAffine(frame, win_M, (awW, awH), flags=cv2.INTER_LINEAR),
                                           cv2.COLOR_BGR2GRAY).astype(np.float32)
                res = cv2.matchTemplate(aligned, anchor_tmpl, cv2.TM_CCOEFF_NORMED)
                _, mx, _, mloc = cv2.minMaxLoc(res)
                if mx >= args.anchor_min_cc:
                    px, py = mloc
                    # Sub-pixel parabolic peak refinement.
                    if 0 < px < res.shape[1] - 1:
                        dl, dc, dr = res[py, px - 1], res[py, px], res[py, px + 1]
                        den = dl - 2 * dc + dr
                        if abs(den) > 1e-6:
                            px = px + 0.5 * (dl - dr) / den
                    if 0 < py < res.shape[0] - 1:
                        du, dc, dd = res[py - 1, mloc[0]], res[mloc[1], mloc[0]], res[py + 1, mloc[0]]
                        den = du - 2 * dc + dd
                        if abs(den) > 1e-6:
                            py = py + 0.5 * (du - dd) / den
                    # Residual: where the knot landed vs where it should be (frame-0).
                    dx = px - exp_x; dy = py - exp_y
                    Tcorr = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], dtype=np.float64)
                    M3 = Tcorr @ M3
                    anchor["n_ok"] += 1
                else:
                    anchor["n_skip"] += 1
            out_M = (T_out @ M3)[:2].astype(np.float32)
            stab = cv2.warpAffine(frame, out_M, (out_w, out_h),
                                  flags=cv2.INTER_CUBIC, borderMode=border_mode)
            proc.stdin.write(stab.tobytes())
            if fi % 200 == 0:
                extra = f" refine_ok={refine['n_ok']} skip={refine['n_skip']}" if refine else ""
                print(f"  render {fi}/{N}{extra}", flush=True)
        if dec is not None:
            dec.stdout.close(); dec.wait()
        else:
            cap.release()
        proc.stdin.close()
        proc.wait()
        if refine:
            print(f"refine: {refine['n_ok']} ok, {refine['n_skip']} skipped", flush=True)
        if anchor:
            print(f"anchor: {anchor['n_ok']} ok, {anchor['n_skip']} skipped", flush=True)
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
