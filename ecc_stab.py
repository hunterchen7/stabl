"""Template-ECC single-patch stabilization with auto-crop.

Pick a small rectangle of pixels (e.g. a textured chunk of the perch stick).
For every frame, ECC image-alignment finds the transform (translation +
rotation, optionally + scale) that maps that patch back to its frame-0 pose,
then the whole frame is warped by it. Only pixels inside the patch drive the
fit, so the moving bird never pulls it.

Two improvements over a naive whole-frame ECC:
  * Windowed, full-res solve. We crop a search window around the patch and run
    ECC there at (near) full resolution, so the patch has hundreds of pixels
    instead of ~9 after a global downscale — that's what makes the lock tight
    instead of mushy.
  * Two-pass auto-crop. Pass 1 records every frame's warp; we compute the
    largest border-safe rectangle (no black edges ever revealed) and frame it
    to the requested aspect, biased to include the subject. Pass 2 renders.

Usage:
  python ecc_stab.py --input clip.mp4 --output stab.mp4 \
      --template_bbox x1,y1,x2,y2 [--motion euclidean|affine] [--auto_crop] \
      [--aspect 16:9] [--bias_y -200] [--draw_bbox]
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
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
}


def to3(m):
    return np.vstack([m, [0, 0, 1]]).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--template_bbox", required=True, help="x1,y1,x2,y2 in SOURCE px.")
    ap.add_argument("--motion", choices=list(MOTIONS), default="euclidean")
    ap.add_argument("--ecc_max_dim", type=int, default=1280,
                    help="Downscale frames so max dim <= this for the ECC solve. Higher = "
                         "more patch pixels = tighter lock, but slower. Whole frame is used "
                         "(masked to the patch) so capture range stays large.")
    ap.add_argument("--mask_pad", type=int, default=80,
                    help="Pad the patch rectangle by this many px when masking, so ECC has "
                         "enough textured stick to lock on.")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--min_cc", type=float, default=0.5,
                    help="Below this ECC correlation, hold the last good transform.")
    ap.add_argument("--auto_crop", action="store_true",
                    help="Two-pass: size the crop to the largest border-safe rectangle.")
    ap.add_argument("--aspect", default="16:9", help="Output aspect for auto_crop, e.g. 16:9.")
    ap.add_argument("--bias_y", type=int, default=0,
                    help="Shift the auto_crop window vertically (negative = up, toward the "
                         "bird) within the border-safe region.")
    ap.add_argument("--bias_x", type=int, default=0)
    ap.add_argument("--crop_w", type=int, default=0, help="Manual crop (ignored if --auto_crop).")
    ap.add_argument("--crop_h", type=int, default=0)
    ap.add_argument("--offset_x", type=int, default=0)
    ap.add_argument("--offset_y", type=int, default=0)
    ap.add_argument("--bitrate", default="12M")
    ap.add_argument("--draw_bbox", action="store_true")
    args = ap.parse_args()

    x1, y1, x2, y2 = (int(v) for v in args.template_bbox.split(","))
    cap = cv2.VideoCapture(args.input)
    ok, frame0 = cap.read()
    if not ok:
        sys.exit("cannot read frame 0")
    H, W = frame0.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Whole-frame ECC at a moderate downscale. The whole frame gives ECC a large
    # capture range (so it stays locked even as the camera drifts far from frame 0),
    # while the mask restricts the actual fit to the patch. Higher ecc_max_dim =
    # more patch pixels = tighter sub-pixel lock.
    f = min(1.0, args.ecc_max_dim / max(W, H))
    Ws, Hs = int(W * f), int(H * f)
    gray0_s = cv2.resize(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0, (Ws, Hs))

    # Mask = patch padded a little, in downscaled coords. Some pad gives ECC enough
    # textured stick to lock on; too much would let the bird leak into the fit.
    pad = args.mask_pad
    mask_s = np.zeros((Hs, Ws), np.uint8)
    cv2.rectangle(mask_s,
                  (int(max(0, x1 - pad) * f), int(max(0, y1 - pad) * f)),
                  (int(min(W, x2 + pad) * f), int(min(H, y2 + pad) * f)), 255, -1)

    mtype = MOTIONS[args.motion]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, args.iterations, args.eps)
    print(f"src {W}x{H}@{fps:.1f} {N}f | ecc@{Ws}x{Hs} (f={f:.3f}) | "
          f"patch ({x1},{y1})-({x2},{y2}) pad={pad} motion={args.motion}", flush=True)

    # ---- Pass 1: ECC every frame, record full-frame warps (frame0->current). ----
    warps = []          # 2x3 float32, maps frame0-full -> current-full
    last_good = np.eye(2, 3, dtype=np.float32)   # downscaled coords; never corrupted
    n_fail = 0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi == 0:
            warps.append(np.eye(2, 3, dtype=np.float32))
            fi += 1
            continue
        gray_s = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0, (Ws, Hs))

        def try_ecc(warm):
            w = warm.copy()   # fresh copy — a thrown ECC call can leave NaN
            try:
                c, w = cv2.findTransformECC(gray0_s, gray_s, w, mtype, criteria, mask_s, 5)
                if np.isfinite(w).all() and c >= args.min_cc:
                    return w, c, True
            except cv2.error:
                pass
            return None, -1.0, False

        # Try the warm-start first; on failure, re-seed from identity so a few
        # bad frames can't freeze the warm-start and cascade forever.
        warp_try, cc, good = try_ecc(last_good)
        if not good:
            warp_try, cc, good = try_ecc(np.eye(2, 3, dtype=np.float32))
        if good:
            last_good = warp_try.copy()
        else:
            n_fail += 1
        # downscaled -> full-res: linear (rotation) part unchanged, translation /f.
        W_full = last_good.copy()
        W_full[:, 2] = W_full[:, 2] / f
        warps.append(W_full.astype(np.float32))
        if fi % 200 == 0:
            print(f"  pass1 {fi}/{N} cc={'HOLD' if not good else f'{cc:.4f}'} fails={n_fail}", flush=True)
        fi += 1
    cap.release()
    Nf = len(warps)
    print(f"pass1 done: {Nf} frames, {n_fail} holds", flush=True)

    # ---- Border-safe region from the warps (in stabilized/frame0 space). ----
    # dst pixel p is valid iff W_full @ p is inside the source rect, so the valid
    # region in dst space is W_full^-1 applied to the source rectangle.
    src_corners = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], dtype=np.float64).T
    safe_l, safe_t = -1e9, -1e9
    safe_r, safe_b = 1e9, 1e9
    for W_full in warps:
        inv = np.linalg.inv(to3(W_full))
        q = inv @ src_corners            # 3x4, columns = TL,TR,BR,BL in dst space
        q = (q[:2] / q[2]).T             # 4x2
        tl, tr, br, bl = q
        safe_l = max(safe_l, tl[0], bl[0])
        safe_r = min(safe_r, tr[0], br[0])
        safe_t = max(safe_t, tl[1], tr[1])
        safe_b = min(safe_b, bl[1], br[1])
    safe_l = max(0, int(np.ceil(safe_l))); safe_t = max(0, int(np.ceil(safe_t)))
    safe_r = min(W, int(np.floor(safe_r))); safe_b = min(H, int(np.floor(safe_b)))
    print(f"border-safe region: x[{safe_l},{safe_r}] y[{safe_t},{safe_b}] "
          f"= {safe_r - safe_l}x{safe_b - safe_t}", flush=True)

    if args.auto_crop and (safe_r - safe_l < 64 or safe_b - safe_t < 64):
        print("WARNING: border-safe region collapsed (tracking likely failed on many "
              "frames). Falling back to full frame — fix tracking before trusting output.",
              flush=True)
        safe_l, safe_t, safe_r, safe_b = 0, 0, W, H

    if args.auto_crop:
        aw, ah = (int(v) for v in args.aspect.split(":"))
        avail_w = safe_r - safe_l
        avail_h = safe_b - safe_t
        # Largest aspect-correct rect inside the safe region.
        if avail_w / avail_h > aw / ah:
            crop_h = avail_h
            crop_w = int(crop_h * aw / ah)
        else:
            crop_w = avail_w
            crop_h = int(crop_w * ah / aw)
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2
        # Center in the safe region, then apply bias, clamped to stay safe.
        cx = (safe_l + safe_r) / 2 + args.bias_x
        cy = (safe_t + safe_b) / 2 + args.bias_y
        cx0 = int(round(cx - crop_w / 2))
        cy0 = int(round(cy - crop_h / 2))
        cx0 = max(safe_l, min(cx0, safe_r - crop_w))
        cy0 = max(safe_t, min(cy0, safe_b - crop_h))
    else:
        crop_w = args.crop_w or W
        crop_h = args.crop_h or H
        bcx = (x1 + x2) / 2 + args.offset_x
        bcy = (y1 + y2) / 2 + args.offset_y
        cx0 = int(bcx - crop_w / 2)
        cy0 = int(bcy - crop_h / 2)
    print(f"crop: {crop_w}x{crop_h} at ({cx0},{cy0})", flush=True)

    # ---- Pass 2: re-read, warp by stored matrices, crop, encode. ----
    crop_M = np.float32([[1, 0, -cx0], [0, 1, -cy0]])
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

    cap = cv2.VideoCapture(args.input)
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or fi >= Nf:
            break
        W_full = warps[fi]
        aligned = cv2.warpAffine(frame, W_full, (W, H),
                                 flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT)
        if args.draw_bbox:
            cv2.rectangle(aligned, (x1, y1), (x2, y2), (0, 255, 0), 3)
        out = cv2.warpAffine(aligned, crop_M, (crop_w, crop_h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        proc.stdin.write(out.tobytes())
        if fi % 200 == 0:
            print(f"  pass2 {fi}/{Nf}", flush=True)
        fi += 1
    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"done -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
