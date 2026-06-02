"""After DLC H5 files are pulled back from Olares, extract per-clip tracks and
run stabl with --auto_crop for each. Outputs land in ~/.stabl-workspace/<shoot>/<clip>/<clip>_dlc_eye.mp4.

Usage:
  .venv-dlc/bin/python batch_stabl.py <shoot_name> <h5_dir> <source_dir>
  e.g. batch_stabl.py 10160524 /tmp/dlc_h5_pulled /Users/hunterchen/Pictures/2026/10160524
"""
import sys
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import numpy as np


HEAD_CLUSTER = [
    "left_eye", "right_eye", "crown", "forehead", "bill",
    "upper_cere", "lower_cere", "chin",
]
WORKSPACE = Path.home() / ".stabl-workspace"
STABL_VENV = Path.home() / "Documents/GitHub/stabl/.venv/bin/python"
STABL_PY = Path.home() / "Documents/GitHub/stabl/stabl.py"


def extract_track(h5_path, csv_path):
    df = pd.read_hdf(h5_path)
    scorer = df.columns.get_level_values(0).unique()[0]
    a = df[scorer]["animal0"]
    available = list(a.columns.get_level_values(0).unique())
    parts = [p for p in HEAD_CLUSTER if p in available]
    rows = []
    for i in range(len(a)):
        xs, ys, ws = [], [], []
        for p in parts:
            x = a[p]["x"].iloc[i]; y = a[p]["y"].iloc[i]; l = a[p]["likelihood"].iloc[i]
            if l >= 0.5 and x > 0 and y > 0:
                xs.append(x); ys.append(y); ws.append(l)
        if xs:
            xs = np.array(xs); ys = np.array(ys); ws = np.array(ws)
            rows.append((i, float(np.average(xs, weights=ws)),
                         float(np.average(ys, weights=ws)), float(ws.mean())))
        else:
            rows.append((i, -1.0, -1.0, 0.0))
    with open(csv_path, "w") as f:
        f.write("frame,x,y,confidence\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]:.3f}\n")
    valid = [r for r in rows if r[1] >= 0]
    return len(rows), len(valid)


def main():
    shoot_name = sys.argv[1]
    h5_dir = Path(sys.argv[2])
    src_dir = Path(sys.argv[3])

    h5s = sorted(h5_dir.glob("C*_superanimal_bird_*.h5"))
    print(f"Found {len(h5s)} H5 files")

    for h5 in h5s:
        clip_id = h5.name.split("_superanimal_bird")[0]
        src = src_dir / f"{clip_id}.MP4"
        if not src.exists():
            print(f"SKIP {clip_id}: no source at {src}")
            continue
        ws = WORKSPACE / shoot_name / clip_id
        ws.mkdir(parents=True, exist_ok=True)
        csv = ws / f"{clip_id}_track.csv"
        n_total, n_valid = extract_track(h5, csv)
        print(f"  {clip_id} track: {n_valid}/{n_total} valid")

        out = ws / f"{clip_id}_dlc_eye.mp4"
        cmd = [str(STABL_VENV), str(STABL_PY), str(src), str(out),
               "--track_csv", str(csv),
               "--auto_crop",
               "--auto_crop_percentile", "2",
               "--smoothing_window", "3",
               "--max_pixel_shift", "200",
               "--bitrate", "50M"]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  {clip_id} stabl FAILED: {r.stderr[-400:]}")
        else:
            # Pull the auto-crop dims out of stdout
            dims = "?"
            for line in r.stdout.splitlines():
                if "optimal 16:9 crop" in line:
                    dims = line.split("=")[-1].strip()
            elapsed = time.time() - t0
            print(f"  {clip_id} stabl OK in {elapsed:.0f}s: {dims}, "
                  f"{out.stat().st_size/1e6:.0f}MB")


if __name__ == "__main__":
    main()
