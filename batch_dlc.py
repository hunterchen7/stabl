"""Run DLC SuperAnimal-Bird on a list of clips, then stabilize each with --track_csv.

Sequential because DLC inference is MPS-bound; running clips back-to-back avoids
re-importing torch/deeplabcut each clip.

Usage:
  .venv-dlc/bin/python batch_dlc.py <shoot_dir> <clip_id> [clip_id ...]
  e.g.  batch_dlc.py /Users/hunterchen/Pictures/2026/10160524 C0062 C0063 ...
"""
import sys
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
import numpy as np
import deeplabcut


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
            if l >= 0.3 and x > 0 and y > 0:
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
    shoot_dir = Path(sys.argv[1])
    clip_ids = sys.argv[2:]
    shoot_name = shoot_dir.name
    print(f"Shoot: {shoot_dir}  Clips: {clip_ids}")

    for clip_id in clip_ids:
        t0 = time.time()
        src = shoot_dir / f"{clip_id}.MP4"
        if not src.exists():
            print(f"SKIP {clip_id}: source not found {src}")
            continue
        ws = WORKSPACE / shoot_name / clip_id
        ws.mkdir(parents=True, exist_ok=True)
        tmp = Path("/tmp") / f"{clip_id}_dlc_input.mp4"
        if not tmp.exists():
            shutil.copy(src, tmp)
        print(f"\n=== {clip_id} ({src.stat().st_size/1e6:.0f}MB) ===")

        # DLC inference
        result = deeplabcut.video_inference_superanimal(
            videos=[str(tmp)],
            superanimal_name="superanimal_bird",
            model_name="resnet_50",
            detector_name="fasterrcnn_mobilenet_v3_large_fpn",
            video_adapt=False,
            device="mps",
        )
        base = tmp.stem
        h5 = Path(f"/tmp/{base}_superanimal_bird_resnet_50_fasterrcnn_mobilenet_v3_large_fpn.h5")
        if not h5.exists():
            print(f"FAIL {clip_id}: no H5 produced")
            continue
        csv = ws / f"{clip_id}_track.csv"
        n_total, n_valid = extract_track(h5, csv)
        print(f"  track: {n_valid}/{n_total} valid")

        # stabl: auto_crop, y_offset=300, smoothing 3
        out = ws / f"{clip_id}_dlc_eye.mp4"
        cmd = [str(STABL_VENV), str(STABL_PY), str(tmp), str(out),
               "--track_csv", str(csv),
               "--auto_crop",
               "--color_offset_y", "300",
               "--smoothing_window", "3",
               "--max_pixel_shift", "200",
               "--bitrate", "50M"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  stabl failed: {r.stderr[-500:]}")
        else:
            print(f"  stabl OK: {out.stat().st_size/1e6:.0f}MB")

        elapsed = time.time() - t0
        print(f"  done in {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
