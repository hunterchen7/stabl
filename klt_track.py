"""Local full-res KLT tracker with frame-0-referenced fitting + re-seeding.

Fully CPU. The key to a drift-free lock: every frame's warp is fit from the
tracked features' CURRENT positions back to their FRAME-0 reference positions
(not frame-to-frame, which accumulates drift). Features that die are replaced
by re-detecting fresh ones and assigning each a frame-0 reference via the
current warp — so the feature pool never runs out (the failure mode of plain
fixed-feature KLT) while the fit stays anchored to frame 0.

Output warps.json: {width,height,fps,n_frames, warps:[[2x3],...]} each mapping
frame_i -> frame_0. Feed to klt_affine.py --warps_json (auto-crop, optional
ECC --refine_bbox for a final sub-pixel polish, render).

Usage:
  python klt_track.py --input clip.mp4 --output_json warps.json \
      --feature_bbox x1,y1,x2,y2 [--n_points 100]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def similarity_fit(src, dst, rigid=False):
    """Fit src->dst. rigid=False: rotation + uniform scale + translation.
    rigid=True: rotation + translation only (scale forced to 1) — prevents
    apparent zoom/resizing when the feature set gets disturbed."""
    src_m = src.mean(0); dst_m = dst.mean(0)
    sc = src - src_m; dc = dst - dst_m
    Hc = (sc.T @ dc) / max(len(src), 1)
    U, S, Vt = np.linalg.svd(Hc)
    d = 1.0 if np.linalg.det(Vt.T @ U.T) > 0 else -1.0
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    if rigid:
        s = 1.0
    else:
        var = (sc ** 2).sum() / max(len(src), 1)
        s = (S @ np.array([1.0, d])) / var if var > 1e-9 else 1.0
    t = dst_m - s * (R @ src_m)
    M = np.eye(3); M[:2, :2] = s * R; M[:2, 2] = t
    return M


def robust_fit(src, dst, rigid=False):
    M = similarity_fit(src, dst, rigid)
    proj = (M[:2, :2] @ src.T).T + M[:2, 2]
    res = np.linalg.norm(proj - dst, axis=1)
    keep = res <= max(3.0 * np.median(res), 1.0)
    if 3 <= keep.sum() < len(src):
        M = similarity_fit(src[keep], dst[keep], rigid)
        proj = (M[:2, :2] @ src.T).T + M[:2, 2]
        res = np.linalg.norm(proj - dst, axis=1)
    return M, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--feature_bbox", required=True, help="x1,y1,x2,y2 SOURCE px.")
    ap.add_argument("--n_points", type=int, default=100)
    ap.add_argument("--win", type=int, default=31)
    ap.add_argument("--max_level", type=int, default=4)
    ap.add_argument("--bidir_thresh", type=float, default=1.0)
    ap.add_argument("--max_resid", type=float, default=2.0,
                    help="Drop a feature whose fit residual exceeds this (px).")
    ap.add_argument("--reseed_min", type=int, default=30,
                    help="Re-detect features when the live count drops below this.")
    ap.add_argument("--rigid", action="store_true",
                    help="Rotation+translation only (no scale) — prevents apparent zoom.")
    ap.add_argument("--max_jump", type=float, default=40.0,
                    help="If the fitted warp's translation jumps more than this (px) from "
                         "the previous frame, treat as wing-capture and HOLD.")
    ap.add_argument("--min_fit", type=int, default=6,
                    help="If fewer than this many features survive LK, HOLD.")
    ap.add_argument("--bootstrap", type=int, default=8,
                    help="Accept the first this-many frames unconditionally (seed velocity).")
    args = ap.parse_args()

    bx1, by1, bx2, by2 = (int(v) for v in args.feature_bbox.split(","))
    cap = cv2.VideoCapture(args.input)
    ok, frame0 = cap.read()
    if not ok:
        sys.exit("cannot read frame 0")
    H, W = frame0.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lk = dict(winSize=(args.win, args.win), maxLevel=args.max_level,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def detect_in_stick(gray, A):
        """Detect features in the stick bbox mapped into the current frame
        (A maps cur->0, so frame-0 bbox maps to current via A^-1)."""
        A0 = np.linalg.inv(A)
        c = np.array([[bx1, by1, 1], [bx2, by1, 1], [bx2, by2, 1], [bx1, by2, 1]]).T
        q = (A0 @ c)[:2].T
        m = np.zeros((H, W), np.uint8)
        m[max(0, int(q[:, 1].min())):min(H, int(q[:, 1].max())),
          max(0, int(q[:, 0].min())):min(W, int(q[:, 0].max()))] = 255
        return cv2.goodFeaturesToTrack(gray, maxCorners=args.n_points, qualityLevel=0.01,
                                       minDistance=10, mask=m, blockSize=9)

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    p = detect_in_stick(gray0, np.eye(3))
    if p is None or len(p) < 4:
        sys.exit(f"too few features ({0 if p is None else len(p)})")
    cur = p.reshape(-1, 2).astype(np.float32)   # current-frame positions
    ref = cur.copy()                            # frame-0 reference positions
    print(f"src {W}x{H}@{fps:.1f} {N}f | frame-0-referenced KLT in "
          f"({bx1},{by1})-({bx2},{by2}), {len(cur)} seed features", flush=True)

    A = np.eye(3)
    A_prev = np.eye(3)              # for constant-velocity prediction
    warps = [A[:2].astype(np.float32).copy()]
    prev_gray = gray0
    n_reseed = 0
    n_held = 0
    for fi in range(1, N):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Replenish features BEFORE tracking so the pool never empties.
        if len(cur) < args.reseed_min:
            p = detect_in_stick(prev_gray, A)
            if p is not None and len(p) >= 4:
                newc = p.reshape(-1, 2).astype(np.float32)
                pr = (A[:2, :2] @ newc.T).T + A[:2, 2]
                inb = (pr[:, 0] >= bx1) & (pr[:, 0] <= bx2) & (pr[:, 1] >= by1) & (pr[:, 1] <= by2)
                if inb.sum() >= 4:
                    cur = np.vstack([cur, newc[inb]]) if len(cur) else newc[inb]
                    ref = np.vstack([ref, pr[inb]]) if len(ref) else pr[inb]
                    n_reseed += 1
        if len(cur) < 3:
            warps.append(A[:2].astype(np.float32).copy()); prev_gray = gray; n_held += 1
            continue
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, cur.reshape(-1, 1, 2), None, **lk)
        back, _, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **lk)
        fb = np.linalg.norm((back.reshape(-1, 2) - cur), axis=1)
        st = st.flatten()
        good = (st == 1) & (fb < args.bidir_thresh)
        nxt = nxt.reshape(-1, 2)

        # Simple: fit from LK-survivors (drift-free, frame-0 referenced). Occlusion
        # spikes are repaired afterward by the post-process deglitch, not here.
        if good.sum() >= 3:
            M, res = robust_fit(nxt[good], ref[good], args.rigid)   # cur -> frame0
            A = M
            keep = good.copy()
            gi = np.where(good)[0]
            keep[gi[res > args.max_resid]] = False
            cur = nxt[keep]; ref = ref[keep]
        else:
            cur = nxt[good]; ref = ref[good]   # A held
        warps.append(A[:2].astype(np.float32).copy())
        prev_gray = gray
        if fi % 200 == 0:
            print(f"  {fi}/{N} feats={len(cur)} reseeds={n_reseed} held={n_held} "
                  f"good={int(good.sum())}", flush=True)

    cap.release()
    out = {"width": W, "height": H, "fps": float(fps), "n_frames": len(warps),
           "warps": [w.tolist() for w in warps]}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f)
    print(f"done -> {args.output_json} ({len(warps)} warps, {n_reseed} re-seeds, {n_held} held)", flush=True)


if __name__ == "__main__":
    main()
