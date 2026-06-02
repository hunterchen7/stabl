"""Extract a weighted (x, y, confidence) track from a DLC H5 using a configurable
list of keypoints. Replaces the head-only logic in dlc_to_track.py.

Presets:
  --preset head   left_eye, right_eye, crown, forehead, bill, upper_cere, lower_cere, chin
  --preset body   back, belly, breast, throat, upper_spine, upper_half_spine,
                  lower_half_spine, left_chest, right_chest
  --preset whole  every available keypoint above min_likelihood

Or pass --parts comma-separated to use exactly those keypoints.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PRESETS = {
    "head": [
        "left_eye", "right_eye", "crown", "forehead", "bill",
        "upper_cere", "lower_cere", "chin",
    ],
    "body": [
        "back", "belly", "breast", "throat",
        "upper_spine", "upper_half_spine", "lower_half_spine",
        "left_chest", "right_chest",
    ],
    "torso_center": [
        "breast", "belly", "back",
        "upper_half_spine", "lower_half_spine",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--preset", choices=list(PRESETS) + ["custom"], default="body")
    ap.add_argument("--parts", default="", help="comma-separated keypoint names (with --preset custom)")
    ap.add_argument("--min_likelihood", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_hdf(args.h5)
    scorer = df.columns.get_level_values(0).unique()[0]
    a = df[scorer]["animal0"]
    available = list(a.columns.get_level_values(0).unique())

    if args.preset == "custom":
        wanted = [p.strip() for p in args.parts.split(",") if p.strip()]
    else:
        wanted = PRESETS[args.preset]
    parts = [p for p in wanted if p in available]
    missing = [p for p in wanted if p not in available]
    if missing:
        print(f"warn: {missing} not in H5; using {parts}", file=sys.stderr)
    if not parts:
        sys.exit("no keypoints matched")

    rows = []
    for i in range(len(a)):
        xs, ys, ws = [], [], []
        for p in parts:
            x = a[p]["x"].iloc[i]; y = a[p]["y"].iloc[i]; l = a[p]["likelihood"].iloc[i]
            if l >= args.min_likelihood and x > 0 and y > 0:
                xs.append(x); ys.append(y); ws.append(l)
        if xs:
            xs = np.array(xs); ys = np.array(ys); ws = np.array(ws)
            rows.append((i,
                         float(np.average(xs, weights=ws)),
                         float(np.average(ys, weights=ws)),
                         float(ws.mean())))
        else:
            rows.append((i, -1.0, -1.0, 0.0))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("frame,x,y,confidence\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]:.3f}\n")
    valid = sum(1 for r in rows if r[1] >= 0)
    print(f"wrote {len(rows)} frames, {valid} valid ({100*valid//max(1,len(rows))}%)", flush=True)


if __name__ == "__main__":
    main()
