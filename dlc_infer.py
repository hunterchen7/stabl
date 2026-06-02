"""DLC SuperAnimal-Bird inference. Wraps deeplabcut.video_inference_superanimal
so it can be called from the server runner as a subprocess. The H5 lands at
--output_h5.

DLC writes its H5 next to the input video file, so we stage the input in /tmp
first to avoid polluting the source dir or the read-only Pictures mount.
"""
import argparse
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_h5", required=True)
    ap.add_argument("--superanimal_name", default="superanimal_bird")
    ap.add_argument("--model_name", default="resnet_50")
    ap.add_argument("--detector_name", default="fasterrcnn_mobilenet_v3_large_fpn")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"no such input: {src}")

    # DLC writes its H5 alongside the video. Stage in /tmp.
    staged = Path("/tmp") / src.name
    if staged != src:
        shutil.copy(src, staged)

    print(f"DLC: starting on {staged}", flush=True)
    import deeplabcut
    deeplabcut.video_inference_superanimal(
        videos=[str(staged)],
        superanimal_name=args.superanimal_name,
        model_name=args.model_name,
        detector_name=args.detector_name,
        video_adapt=False,
        device=args.device,
    )

    expected = (
        Path("/tmp")
        / f"{staged.stem}_{args.superanimal_name}_{args.model_name}_{args.detector_name}.h5"
    )
    if not expected.exists():
        sys.exit(f"DLC: H5 not produced at {expected}")
    Path(args.output_h5).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(expected), args.output_h5)
    if staged != src:
        try:
            staged.unlink()
        except OSError:
            pass
    print(f"DLC: H5 -> {args.output_h5}", flush=True)


if __name__ == "__main__":
    main()
