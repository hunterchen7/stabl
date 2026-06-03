"""Track N points across a video using CoTracker3 (Meta, 2024).

Unlike KLT, CoTracker3 jointly tracks all points using a transformer over a
sliding temporal window, predicts per-frame visibility, and can re-acquire
features after occlusion. Designed for the "wing flap covers the feature for
5 frames and then it comes back" case that breaks KLT.

The model is downscaled-input only (trained on ~512px). We downscale, track,
then scale coords back to source resolution.

Output JSON:
{
  "width": <src_w>, "height": <src_h>, "fps": <fps>, "n_frames": <N>,
  "n_points": <P>, "scale": <track_w / src_w>,
  "tracks": [[[x, y, vis], ...P points], ...N frames]
}
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--n_points", type=int, default=80,
                    help="Number of points to track (more = better stat for RANSAC).")
    ap.add_argument("--max_track_dim", type=int, default=640,
                    help="Downscale source so max(W,H) is at most this for tracking.")
    ap.add_argument("--mode", choices=["offline", "online"], default="offline")
    ap.add_argument("--mask_circle", default=None,
                    help="cx,cy,r in SOURCE coords — exclude this circle from query "
                         "point selection (e.g. mask the bird).")
    ap.add_argument("--query_points", default=None,
                    help="Semicolon-separated explicit query coords in SOURCE resolution: "
                         "'x1,y1;x2,y2;...'. Skips goodFeaturesToTrack and tracks exactly "
                         "these points instead. For manual selection.")
    ap.add_argument("--expand_patch", type=int, default=0,
                    help="If >0, each query point gets expanded into a 3x3 grid of points "
                         "with this radius (pixels in source). Lets a single user-picked "
                         "location yield rotation/scale info via relative motion.")
    ap.add_argument("--track_window", default=None,
                    help="x1,y1,x2,y2 in SOURCE px — crop to this window and track it at "
                         "(near) full resolution. Much more precise than downscaling the "
                         "whole 4K frame. If omitted but query_points given, derived from "
                         "their bbox + track_window_pad.")
    ap.add_argument("--track_window_pad", type=int, default=300,
                    help="Pad (px) for the auto-derived track window; must exceed camera "
                         "shake so the points never leave the window.")
    args = ap.parse_args()

    import torch
    print("loading cotracker3...", flush=True)
    model_name = "cotracker3_offline" if args.mode == "offline" else "cotracker3_online"
    model = torch.hub.load("facebookresearch/co-tracker", model_name, trust_repo=True).cuda()
    model.eval()

    # Load video
    cap = cv2.VideoCapture(args.input)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Optional crop window: track only a small region (around the stick) at full
    # resolution. Far more precise than downscaling the whole 4K frame, because
    # the tracked points aren't scaled up (a 0.5px error at 640px becomes ~6px at
    # 4K). Window is given, or derived from the query points + pad.
    ox, oy = 0, 0
    win_w, win_h = W, H
    if args.track_window:
        ox, oy, wx2, wy2 = (int(v) for v in args.track_window.split(","))
        win_w, win_h = wx2 - ox, wy2 - oy
    elif args.query_points:
        qp = np.array([[float(x), float(y)] for x, y in
                       (c.split(",") for c in args.query_points.split(";"))])
        pad = args.track_window_pad + (args.expand_patch if args.expand_patch > 0 else 0)
        ox = max(0, int(qp[:, 0].min() - pad)); oy = max(0, int(qp[:, 1].min() - pad))
        wx2 = min(W, int(qp[:, 0].max() + pad)); wy2 = min(H, int(qp[:, 1].max() + pad))
        win_w, win_h = wx2 - ox, wy2 - oy

    scale = min(1.0, args.max_track_dim / max(win_w, win_h))
    Wt, Ht = int(win_w * scale), int(win_h * scale)
    print(f"src {W}x{H} @ {fps:.2f}fps, {N} frames | window ({ox},{oy}) {win_w}x{win_h} "
          f"-> tracking at {Wt}x{Ht} (scale {scale:.3f})", flush=True)

    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        f = f[oy:oy + win_h, ox:ox + win_w]
        if scale != 1.0:
            f = cv2.resize(f, (Wt, Ht))
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    video = np.stack(frames)  # [N, Ht, Wt, 3]
    print(f"loaded {len(frames)} frames, {video.nbytes / 1e9:.2f} GB", flush=True)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
    video = video[None].cuda()  # [1, N, 3, Ht, Wt]

    # Pick query points on frame 0 (in tracking coords = (src - window_origin) * scale).
    if args.query_points:
        pts_src = np.array([
            [float(x), float(y)]
            for x, y in (chunk.split(",") for chunk in args.query_points.split(";"))
        ])
        if args.expand_patch > 0:
            r = args.expand_patch
            grid = []
            for cx, cy in pts_src:
                for dy in (-r, 0, r):
                    for dx in (-r, 0, r):
                        grid.append([cx + dx, cy + dy])
            pts_src = np.array(grid)
            print(f"expanded {len(pts_src)//9} centers into {len(pts_src)} grid points (r={r})", flush=True)
        pts = (pts_src - np.array([ox, oy])) * scale  # src -> window-local tracking coords
        P = len(pts)
        queries = torch.zeros(1, P, 3, device="cuda")
        queries[0, :, 0] = 0
        queries[0, :, 1] = torch.from_numpy(pts[:, 0].astype(np.float32)).cuda()
        queries[0, :, 2] = torch.from_numpy(pts[:, 1].astype(np.float32)).cuda()
        print(f"using {P} caller-provided query points", flush=True)
    else:
        gray0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        mask = None
        if args.mask_circle:
            cx, cy, r = (float(v) for v in args.mask_circle.split(","))
            cx = (cx - ox) * scale; cy = (cy - oy) * scale; r *= scale
            mask = np.ones((Ht, Wt), dtype=np.uint8) * 255
            cv2.circle(mask, (int(cx), int(cy)), int(r), 0, -1)
        corners = cv2.goodFeaturesToTrack(
            gray0, maxCorners=args.n_points, qualityLevel=0.005,
            minDistance=12, mask=mask, blockSize=9)
        if corners is None or len(corners) < 3:
            sys.exit(f"too few query points found ({0 if corners is None else len(corners)})")
        P = len(corners)
        queries = torch.zeros(1, P, 3, device="cuda")
        queries[0, :, 0] = 0
        queries[0, :, 1] = torch.from_numpy(corners[:, 0, 0]).cuda()
        queries[0, :, 2] = torch.from_numpy(corners[:, 0, 1]).cuda()
        print(f"picked {P} query points; running {args.mode} inference", flush=True)

    with torch.no_grad():
        if args.mode == "offline":
            pred_tracks, pred_visibility = model(video, queries=queries)
        else:
            model(video_chunk=video[:, : model.step * 2], is_first_step=True, queries=queries)
            for ind in range(0, video.shape[1] - model.step, model.step):
                pred_tracks, pred_visibility = model(
                    video_chunk=video[:, ind : ind + model.step * 2]
                )

    pred_tracks = pred_tracks.cpu().numpy()[0]      # [N, P, 2] in tracking coords
    pred_visibility = pred_visibility.cpu().numpy()[0]  # [N, P]
    print(f"got tracks {pred_tracks.shape}, mean visibility {pred_visibility.mean():.3f}", flush=True)

    # Window-local tracking coords -> full-frame source coords.
    src_tracks = pred_tracks / scale + np.array([ox, oy], dtype=np.float32)
    out = {
        "width": W, "height": H, "fps": float(fps),
        "n_frames": int(src_tracks.shape[0]), "n_points": int(P),
        "scale": float(scale),
        "tracks": [
            [[float(src_tracks[fi, p, 0]),
              float(src_tracks[fi, p, 1]),
              float(pred_visibility[fi, p])]
             for p in range(P)]
            for fi in range(src_tracks.shape[0])
        ],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f)
    print(f"done -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
