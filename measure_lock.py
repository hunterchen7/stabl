"""Objectively measure how locked a region is in a stabilized video.

Phase-correlates a chosen rectangle of every Nth frame against the same
rectangle of frame 0 and reports the displacement statistics. A perfect lock
reads ~0.0px everywhere; visible shake shows up as px-level displacements.

Usage:
  python measure_lock.py --input stab.mp4 --bbox x1,y1,x2,y2 [--step 5]
"""
import argparse

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--bbox", required=True, help="x1,y1,x2,y2 in OUTPUT-video px")
    ap.add_argument("--step", type=int, default=5, help="measure every Nth frame")
    args = ap.parse_args()

    x1, y1, x2, y2 = (int(v) for v in args.bbox.split(","))
    cap = cv2.VideoCapture(args.input)
    ok, f0 = cap.read()
    if not ok:
        raise SystemExit("cannot read frame 0")
    H, W = f0.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    win = cv2.createHanningWindow((x2 - x1, y2 - y1), cv2.CV_32F)
    ref = cv2.cvtColor(f0[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)

    disps = []
    fi = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fi += 1
        if fi % args.step:
            continue
        cur = cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(ref, cur, win)
        disps.append((fi, dx, dy, resp))
    cap.release()

    arr = np.array([(d[1], d[2]) for d in disps])
    mag = np.linalg.norm(arr, axis=1)
    print(f"frames measured: {len(disps)} (every {args.step})")
    print(f"displacement px  mean={mag.mean():.2f}  median={np.median(mag):.2f}  "
          f"p95={np.percentile(mag, 95):.2f}  max={mag.max():.2f}")
    worst = sorted(disps, key=lambda d: -(d[1] ** 2 + d[2] ** 2))[:5]
    print("worst frames:", [(f, round(dx, 1), round(dy, 1)) for f, dx, dy, _ in worst])


if __name__ == "__main__":
    main()
