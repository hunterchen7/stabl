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

    scale = min(1.0, args.max_track_dim / max(W, H))
    Wt = int(W * scale)
    Ht = int(H * scale)
    print(f"src {W}x{H} @ {fps:.2f}fps, {N} frames -> tracking at {Wt}x{Ht} (scale {scale:.3f})", flush=True)

    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        f = cv2.resize(f, (Wt, Ht))
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    video = np.stack(frames)  # [N, Ht, Wt, 3]
    print(f"loaded {len(frames)} frames, {video.nbytes / 1e9:.2f} GB", flush=True)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
    video = video[None].cuda()  # [1, N, 3, Ht, Wt]

    # Pick query points on frame 0 (in tracking coords).
    if args.query_points:
        pts_src = np.array([
            [float(x), float(y)]
            for x, y in (chunk.split(",") for chunk in args.query_points.split(";"))
        ])
        pts = pts_src * scale  # to tracking coords
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
            cx *= scale; cy *= scale; r *= scale
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

    # Scale back to source
    src_tracks = pred_tracks / scale
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
