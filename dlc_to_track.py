"""Extract a per-frame (x, y) track from a DLC SuperAnimal-Bird H5.

Strategy: take the median position of the head-cluster keypoints, weighted by
likelihood. Falls back gracefully when keypoints drop out.

Usage:
    python dlc_to_track.py input.h5 output.csv [--bodypart left_eye]
"""
import argparse
import sys
import pandas as pd
import numpy as np


HEAD_CLUSTER = [
    "left_eye", "right_eye", "crown", "forehead", "bill",
    "upper_cere", "lower_cere", "chin",
]


def extract_track(h5_path, bodypart=None, min_likelihood=0.3):
    df = pd.read_hdf(h5_path)
    scorer = df.columns.get_level_values(0).unique()[0]
    # Use animal0 (DLC pads to 10 animals; rest are -1)
    a = df[scorer]["animal0"]
    n_frames = len(a)
    parts = [bodypart] if bodypart else HEAD_CLUSTER
    parts = [p for p in parts if p in a.columns.get_level_values(0).unique()]

    rows = []
    for i in range(n_frames):
        xs, ys, ws = [], [], []
        for p in parts:
            x = a[p]["x"].iloc[i]
            y = a[p]["y"].iloc[i]
            l = a[p]["likelihood"].iloc[i]
            if l >= min_likelihood and x > 0 and y > 0:
                xs.append(x); ys.append(y); ws.append(l)
        if xs:
            xs = np.array(xs); ys = np.array(ys); ws = np.array(ws)
            cx = float(np.average(xs, weights=ws))
            cy = float(np.average(ys, weights=ws))
            conf = float(ws.mean())
        else:
            cx = cy = -1.0
            conf = 0.0
        rows.append((i, cx, cy, conf))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("out_csv")
    ap.add_argument("--bodypart", default=None,
                    help="single keypoint (default: weighted head cluster)")
    ap.add_argument("--min_likelihood", type=float, default=0.3)
    args = ap.parse_args()

    rows = extract_track(args.h5, args.bodypart, args.min_likelihood)
    with open(args.out_csv, "w") as f:
        f.write("frame,x,y,confidence\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]:.3f}\n")

    valid = [r for r in rows if r[1] >= 0]
    if valid:
        xs = [r[1] for r in valid]; ys = [r[2] for r in valid]
        print(f"wrote {len(rows)} frames -> {args.out_csv}")
        print(f"  valid: {len(valid)}/{len(rows)} ({100*len(valid)/len(rows):.1f}%)")
        print(f"  x range: [{min(xs):.0f}, {max(xs):.0f}]  shift: {max(xs)-min(xs):.0f}px")
        print(f"  y range: [{min(ys):.0f}, {max(ys):.0f}]  shift: {max(ys)-min(ys):.0f}px")
    else:
        print("no valid detections", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
