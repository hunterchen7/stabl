"""Template-ECC stabilization — lock ONE rigid patch in place.

The user's mental model, implemented literally: pick a rectangle of pixels
(e.g. the perch stick). For every frame, find the geometric transform
(translation + rotation, optionally + scale) that best maps that rectangle
back to where it was in frame 0, then warp the whole frame by it. Because the
alignment only uses pixels *inside the rectangle*, nothing else in the scene
(the bird turning, flapping, flying away; foliage in wind) can ever pull it.

This is OpenCV's findTransformECC (Enhanced Correlation Coefficient) image
alignment, masked to the chosen rectangle. Every frame is aligned to FRAME 0
(not the previous frame), so there is zero drift over long clips.

Usage:
  python ecc_stab.py --input clip.mp4 --output stab.mp4 \
      --template_bbox x1,y1,x2,y2 [--motion euclidean|affine|homography] \
      [--crop_w W --crop_h H --offset_x 0 --offset_y 0] [--draw_bbox]
"""
import argparse
import subprocess
import sys

import cv2
import numpy as np


def pick_encoder(preview: bool = True) -> list[str]:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True).stdout
    if "hevc_videotoolbox" in out:
        return ["-c:v", "hevc_videotoolbox"]
    if "hevc_nvenc" in out:
        return ["-c:v", "hevc_nvenc", "-preset", "p1" if preview else "p4"]
    return ["-c:v", "libx265", "-preset", "ultrafast" if preview else "medium"]


MOTIONS = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,    # translation + rotation
    "affine": cv2.MOTION_AFFINE,          # + scale + shear
    "homography": cv2.MOTION_HOMOGRAPHY,  # + perspective
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--template_bbox", required=True,
                    help="x1,y1,x2,y2 in SOURCE px — the rigid patch to lock (the stick).")
    ap.add_argument("--motion", choices=list(MOTIONS), default="euclidean")
    ap.add_argument("--ecc_max_dim", type=int, default=720,
                    help="Downscale frames so max dim <= this for the ECC solve (speed). "
                         "Warp is scaled back up and applied to full res.")
    ap.add_argument("--mask_pad", type=int, default=300,
                    help="Pad the template rectangle by this many px (source) when masking, "
                         "so camera shake doesn't push the patch out of the mask.")
    ap.add_argument("--min_cc", type=float, default=0.5,
                    help="If ECC correlation drops below this, treat as a failure and hold "
                         "the last good transform instead of trusting a bad fit.")
    ap.add_argument("--crop_w", type=int, default=0, help="0 = full frame")
    ap.add_argument("--crop_h", type=int, default=0)
    ap.add_argument("--offset_x", type=int, default=0,
                    help="Crop center = template center + offset. Use to bring the bird back "
                         "into frame while keeping the stick locked.")
    ap.add_argument("--offset_y", type=int, default=0)
    ap.add_argument("--bitrate", default="12M")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--draw_bbox", action="store_true",
                    help="Draw the locked rectangle on the output (debug — see what's pinned).")
    args = ap.parse_args()

    x1, y1, x2, y2 = (int(v) for v in args.template_bbox.split(","))
    cap = cv2.VideoCapture(args.input)
    ok, frame0 = cap.read()
    if not ok:
        sys.exit("cannot read frame 0")
    H, W = frame0.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
    crop_w = args.crop_w or W
    crop_h = args.crop_h or H
    ref_cx = bcx + args.offset_x
    ref_cy = bcy + args.offset_y

    # Downscale factor for the ECC solve.
    f = min(1.0, args.ecc_max_dim / max(W, H))
    Ws, Hs = int(W * f), int(H * f)
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray0_s = cv2.resize(gray0, (Ws, Hs))

    # Mask = padded template rectangle, in downscaled coords.
    mask_s = np.zeros((Hs, Ws), np.uint8)
    mx1 = int(max(0, x1 - args.mask_pad) * f)
    my1 = int(max(0, y1 - args.mask_pad) * f)
    mx2 = int(min(W, x2 + args.mask_pad) * f)
    my2 = int(min(H, y2 + args.mask_pad) * f)
    cv2.rectangle(mask_s, (mx1, my1), (mx2, my2), 255, -1)
    print(f"src {W}x{H}@{fps:.1f} {N}f | ecc@{Ws}x{Hs} (f={f:.3f}) | "
          f"template ({x1},{y1})-({x2},{y2}) motion={args.motion}", flush=True)

    mtype = MOTIONS[args.motion]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, args.iterations, args.eps)

    enc = pick_encoder()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{crop_w}x{crop_h}", "-r", f"{fps}",
           "-i", "-", "-i", args.input, "-map", "0:v", "-map", "1:a?",
           *enc, "-b:v", args.bitrate, "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
           "-bsf:v", "hevc_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
           "-c:a", "aac", "-b:a", "192k", "-shortest", args.output]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin

    cx0 = int(ref_cx - crop_w / 2)
    cy0 = int(ref_cy - crop_h / 2)
    crop_M = np.float32([[1, 0, -cx0], [0, 1, -cy0]])

    def emit(frame_aligned):
        if args.draw_bbox:
            cv2.rectangle(frame_aligned, (x1, y1), (x2, y2), (0, 255, 0), 4)
        out = cv2.warpAffine(frame_aligned, crop_M, (crop_w, crop_h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        proc.stdin.write(out.tobytes())

    emit(frame0.copy())

    eye = np.eye(3, dtype=np.float32) if mtype == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
    last_good = eye.copy()   # frame->frame0 transform; never corrupted by a failed solve
    n_fail = 0
    for fi in range(1, N):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_s = cv2.resize(gray, (Ws, Hs))
        # Warm-start from the last GOOD warp (a fresh copy — a failed ECC call can
        # leave the passed matrix as NaN, which would otherwise poison every
        # subsequent frame).
        warp_try = last_good.copy()
        failed = False
        try:
            cc, warp_try = cv2.findTransformECC(gray0_s, gray_s, warp_try, mtype, criteria, mask_s, 5)
            if not np.isfinite(warp_try).all() or cc < args.min_cc:
                failed = True
        except cv2.error:
            failed = True
        if failed:
            n_fail += 1
            warp = last_good          # hold last good transform for this frame
        else:
            warp = warp_try
            last_good = warp_try.copy()
        # Scale warp from downscaled coords to full-res.
        if mtype == cv2.MOTION_HOMOGRAPHY:
            S = np.diag([f, f, 1]).astype(np.float32)
            warp_full = np.linalg.inv(S) @ warp @ S
            aligned = cv2.warpPerspective(frame, warp_full, (W, H),
                                          flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                                          borderMode=cv2.BORDER_CONSTANT)
        else:
            warp_full = warp.copy()
            warp_full[:, 2] = warp[:, 2] / f
            aligned = cv2.warpAffine(frame, warp_full, (W, H),
                                     flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT)
        emit(aligned)
        if fi % 100 == 0:
            print(f"  {fi}/{N} cc={'HOLD' if failed else f'{cc:.4f}'} fails={n_fail}", flush=True)

    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"done -> {args.output} ({n_fail} ECC failures)", flush=True)


if __name__ == "__main__":
    main()
