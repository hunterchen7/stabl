import cv2
import argparse
import numpy as np
import subprocess
import threading
from collections import deque


def load_track_csv(path):
    """Read a per-frame track CSV with header 'frame,x,y,confidence' and return
    a dict {frame_index: (x, y)} for rows where x >= 0. Frame indices are
    expected to align with the video's frame ordering (0-based)."""
    track = {}
    with open(path) as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            try:
                frame_idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                continue
            if x >= 0 and y >= 0:
                track[frame_idx] = (int(round(x)), int(round(y)))
    print(f"Loaded {len(track)} tracked frames from {path}")
    return track


def parse_hsv_range(spec):
    """Parse a string like '0,140,90:12,255,255' into ((H,S,V),(H,S,V))."""
    try:
        lo_str, hi_str = spec.split(':')
        lo = tuple(int(x) for x in lo_str.split(','))
        hi = tuple(int(x) for x in hi_str.split(','))
        if len(lo) != 3 or len(hi) != 3:
            raise ValueError
        return (np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Bad --color_range '{spec}'. Expected H,S,V:H,S,V (e.g. 0,140,90:12,255,255)"
        )


def compute_auto_crop_dims(args, frame_width, frame_height, track_csv_points=None):
    """Probe the input video to find the largest 16:9 crop that can be centered
    on the detected (offset-applied) centroid in every frame without ever
    sliding off the source edges. Returns (width, height) or (None, None) on
    failure. Supported in --track_color and --track_csv modes."""
    if not args.track_color and track_csv_points is None:
        print("Auto-crop currently only supported for --track_color / --track_csv modes.")
        return None, None
    xs, ys = [], []
    missed = 0
    if track_csv_points is not None:
        print(f"Auto-crop probe (csv): {len(track_csv_points)} points")
        for (x, y) in track_csv_points.values():
            xs.append(x + args.color_offset_x)
            ys.append(y + args.color_offset_y)
    else:
        print(f"Auto-crop probe: scanning all frames of '{args.input_video}'...")
        cap = cv2.VideoCapture(args.input_video)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            center, _, found = find_color_centroid(
                frame, args.color_range, args.color_min_area, args.color_max_area)
            if found:
                xs.append(center[0] + args.color_offset_x)
                ys.append(center[1] + args.color_offset_y)
            else:
                missed += 1
        cap.release()
    if not xs:
        print("Auto-crop: no centroids detected. Falling back to --width/--height.")
        return None, None
    xs = np.array(xs); ys = np.array(ys)
    # Optional percentile clamp on the centroid range so a handful of outlier
    # detections (e.g. one frame where DLC misfires onto a leaf) don't collapse
    # the crop. With --auto_crop_percentile 2 the crop covers p2..p98 of the
    # centroid range; the remaining ~4% of frames clamp at the source edges
    # at runtime, which the tracker handles fine.
    pct = getattr(args, "auto_crop_percentile", 0) or 0
    if pct > 0:
        xmin = float(np.percentile(xs, pct))
        xmax = float(np.percentile(xs, 100 - pct))
        ymin = float(np.percentile(ys, pct))
        ymax = float(np.percentile(ys, 100 - pct))
    else:
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
    Wc = 2 * min(xmin, frame_width - xmax)
    Hc = 2 * min(ymin, frame_height - ymax)
    aspect = 16 / 9
    if Wc / aspect <= Hc:
        Wc16, Hc16 = Wc, Wc / aspect
    else:
        Wc16, Hc16 = Hc * aspect, Hc
    Wc16 = max(2, int(Wc16) - (int(Wc16) % 2))
    Hc16 = max(2, int(Hc16) - (int(Hc16) % 2))
    print(f"Auto-crop: detected {len(xs)}/{len(xs)+missed} frames, "
          f"centroid X[{int(xs.min())},{int(xs.max())}] Y[{int(ys.min())},{int(ys.max())}] "
          f"-> optimal 16:9 crop = {Wc16}x{Hc16}")
    return Wc16, Hc16


def find_color_centroid(frame, hsv_ranges, min_area, max_area, prev_center=None):
    """
    Find the centroid of the largest connected blob whose pixels fall within
    any of the given HSV ranges and whose area is within [min_area, max_area].
    If prev_center is given and multiple blobs qualify, prefer the one closest
    to prev_center (helps track continuity when multiple colored blobs appear).
    Returns (center_or_None, bbox_or_None, found_bool) where bbox is (x,y,w,h).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in hsv_ranges:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is None:
        return None, None, False
    # Clean small noise + close gaps in the blob
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    num, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for i in range(1, num):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            candidates.append((i, area))
    if not candidates:
        return None, None, False

    if prev_center is not None and len(candidates) > 1:
        prev = np.array(prev_center)
        best = min(candidates,
                   key=lambda c: np.linalg.norm(centroids[c[0]] - prev))[0]
    else:
        best = max(candidates, key=lambda c: c[1])[0]
    cx, cy = centroids[best]
    x, y, w, h, _ = stats[best]
    return (int(cx), int(cy)), (int(x), int(y), int(w), int(h)), True


def autodetect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def autodetect_codec(device):
    if device == 'cuda':
        return 'hevc_nvenc'
    if device == 'mps':
        return 'hevc_videotoolbox'
    return 'libx265'


def reader_thread(pipe, stream_name):
    """A simple thread function to read from a subprocess pipe and print."""
    try:
        for line in iter(pipe.readline, ''):
            print(f"[{stream_name}] {line.strip()}", flush=True)
    finally:
        pipe.close()


def find_best_candidate(boxes, classes, track_ids, target_class_id, frame_width, frame_height):
    """
    Finds the best candidate for tracking in a given frame.
    The 'best' is defined as the largest object of the target class,
    weighted by its proximity to the center of the frame.
    Returns: (best_id, best_center) or (None, None) if no valid candidate is found.
    """
    best_candidate_id = None
    best_center = None
    max_score = 0
    center_of_frame = np.array([frame_width / 2, frame_height / 2])

    for i, box in enumerate(boxes):
        if classes[i] == target_class_id:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            distance_from_center = np.linalg.norm(box_center - center_of_frame)
            score = area / (distance_from_center + 1e-6)

            if score > max_score:
                max_score = score
                best_candidate_id = track_ids[i]
                best_center = (int(box_center[0]), int(box_center[1]))

    return best_candidate_id, best_center


def main(args):
    """
    Main function to process the video stabilization with audio and quality preservation.
    """
    # ... (YOLO model loading and subject validation is the same)
    model = None
    target_class_id = None
    track_csv_points = None
    if args.track_csv:
        track_csv_points = load_track_csv(args.track_csv)
        if not track_csv_points:
            print(f"Error: no valid points loaded from --track_csv {args.track_csv}")
            return
        print(f"CSV-tracking mode: {len(track_csv_points)} pre-computed points")
    elif args.track_color:
        if not args.color_range:
            print("Error: --track_color requires at least one --color_range "
                  "(format: H,S,V:H,S,V).")
            return
        print(f"Color-tracking mode: {len(args.color_range)} HSV range(s), "
              f"area bounds [{args.color_min_area}..{args.color_max_area}]px")
    else:
        from ultralytics import YOLO
        print(f"Loading YOLO model: {args.model} on device: {args.device}...")
        try:
            model = YOLO(args.model)
            model.to(args.device)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return

        # --- Subject Class Validation ---
        target_subject_name = args.target_subject.lower()
        class_names = model.names
        name_to_id = {v.lower(): k for k, v in class_names.items()}
        if target_subject_name not in name_to_id:
            print(
                f"Error: Subject '{args.target_subject}' is not a valid class name.")
            print(f"Available classes are: {list(class_names.values())}")
            return
        target_class_id = name_to_id[target_subject_name]
        print(
            f"Successfully identified target class '{target_subject_name}' with ID: {target_class_id}")

    # --- Video and Audio Setup ---
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.input_video}")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Optional first-pass probe to auto-compute the largest 16:9 crop that
    # can follow the detected centroid without ever clamping.
    if args.auto_crop:
        w, h = compute_auto_crop_dims(
            args, frame_width, frame_height,
            track_csv_points=track_csv_points)
        if w is not None:
            args.width = w
            args.height = h

    # Re-open for the main pass
    cap = cv2.VideoCapture(args.input_video)

    # --- FFmpeg Subprocess Setup ---
    # --bitrate overrides CRF when set. videotoolbox doesn't accept -crf / -preset;
    # map quality to -q:v instead. nvenc uses -cq.
    quality_args = []
    if args.bitrate:
        quality_args = ['-b:v', args.bitrate]
        if 'nvenc' in args.video_codec:
            quality_args = ['-preset', 'slow'] + quality_args
        elif 'videotoolbox' not in args.video_codec:
            quality_args = ['-preset', 'slow'] + quality_args
    elif 'videotoolbox' in args.video_codec:
        # -q:v 0-100, higher = better. Map CRF 0-51 (lower=better) to q 100-0 roughly.
        q = max(1, min(100, int(100 - args.crf * 2)))
        quality_args = ['-q:v', str(q)]
    elif args.video_codec in ('hevc_nvenc', 'h264_nvenc'):
        quality_args = ['-preset', 'slow', '-cq', str(args.crf)]
    else:
        quality_args = ['-preset', 'slow', '-crf', str(args.crf)]

    # QuickTime requires hvc1 tag on HEVC; default hev1 from encoders makes files
    # unplayable in QT/Finder previews even though VLC/ffplay are happy.
    # Also, BGR raw input → HEVC encoders mistag the matrix as GBR/identity →
    # players that assume the missing tag is bt2020 (UHD) produce a green cast.
    # Force bt709 in both the container and the HEVC VUI.
    is_hevc = ('hevc' in args.video_codec or 'x265' in args.video_codec or
               '265' in args.video_codec)
    if is_hevc:
        quality_args = quality_args + [
            '-tag:v', 'hvc1',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-colorspace', 'bt709',
            '-bsf:v',
            'hevc_metadata=video_full_range_flag=0:colour_primaries=1:'
            'transfer_characteristics=1:matrix_coefficients=1',
        ]

    # Force a real BGR (full range) → YUV420p bt709 (TV range) conversion before
    # the encoder. Without this filter, nvenc/x265 treat the BGR raw stream as
    # YUV444 with identity (GBR) matrix coefficients, producing files that play
    # back with a green/magenta cast even when retagged. The scale filter does
    # the matrix conversion explicitly; format=yuv420p picks the pixel format.
    convert_vf = (
        'scale=in_color_matrix=bt709:out_color_matrix=bt709:'
        'in_range=full:out_range=tv:flags=accurate_rnd+full_chroma_int,'
        'format=yuv420p'
    )

    # In visualize mode the pipe carries the full source frame downscaled to 1920x1080
    # with crop box / detection overlay, not the cropped subject window.
    pipe_w = 1920 if args.visualize else args.width
    pipe_h = 1080 if args.visualize else args.height

    ffmpeg_command = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{pipe_w}x{pipe_h}',
        '-r', str(fps), '-i', '-',
        '-i', args.input_video,
        '-map', '0:v:0', '-map', '1:a:0?',
        '-map_metadata', '1', '-movflags', 'use_metadata_tags+faststart',
        '-vf', convert_vf,
        '-c:v', args.video_codec, *quality_args,
        '-c:a', 'copy',
        args.output_video,
    ]
    ffmpeg_process = subprocess.Popen(
        ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    err_thread = threading.Thread(target=reader_thread, args=(
        ffmpeg_process.stderr, "ffmpeg_stderr"), daemon=True)
    err_thread.start()
    print("\nFFmpeg process started. Processing frames...\n")

    # --- Main Processing Loop Initialization ---
    tracked_subject_id = None
    last_known_center = None
    initial_crop_x = max(0, frame_width // 2 - args.width // 2)
    initial_crop_y = max(0, frame_height // 2 - args.height // 2)
    last_crop_coords = {"x1": initial_crop_x, "y1": initial_crop_y,
                        "x2": initial_crop_x + args.width, "y2": initial_crop_y + args.height}
    last_crop_center = (initial_crop_x + args.width // 2,
                        initial_crop_y + args.height // 2)
    center_history = deque(maxlen=args.smoothing_window)
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            subject_found_in_frame = False
            last_known_color_bbox = locals().get('last_known_color_bbox', None)
            if args.track_csv:
                # --- Pre-computed per-frame track (e.g. from DLC keypoints) ---
                pt = track_csv_points.get(frame_count - 1)
                if pt is not None:
                    last_known_center = (pt[0] + args.color_offset_x,
                                         pt[1] + args.color_offset_y)
                    subject_found_in_frame = True
            elif args.track_color:
                # --- Color-anchor detection ---
                center, bbox, found = find_color_centroid(
                    frame, args.color_range,
                    args.color_min_area, args.color_max_area,
                    prev_center=last_known_center,
                )
                if found:
                    # Apply Y offset (positive = shift crop center DOWN relative to the
                    # detected color blob). Useful when the color marker sits on a
                    # specific anatomical region (e.g. the shoulder epaulet on a
                    # red-winged blackbird) but you want the body of the subject
                    # centered, not the marker itself.
                    cx, cy = center
                    last_known_center = (cx + args.color_offset_x,
                                         cy + args.color_offset_y)
                    last_known_color_bbox = bbox
                    subject_found_in_frame = True
            else:
                # --- YOLO Object Detection and Tracking ---
                results = model.track(frame, persist=True,
                                      device=args.device, verbose=False, conf=args.conf)

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    classes = results[0].boxes.cls.int().cpu().tolist()

                    # First, check if our currently tracked subject is still visible.
                    if tracked_subject_id is not None and tracked_subject_id in track_ids:
                        subject_index = track_ids.index(tracked_subject_id)
                        if classes[subject_index] == target_class_id:
                            subject_found_in_frame = True
                            x1, y1, x2, y2 = boxes[subject_index]
                            last_known_center = (
                                int((x1 + x2) / 2) + args.color_offset_x,
                                int((y1 + y2) / 2) + args.color_offset_y)

                    # If we lost the subject, or never had one, find the best new one immediately.
                    if not subject_found_in_frame:
                        old_id = tracked_subject_id
                        best_id, best_center = find_best_candidate(
                            boxes, classes, track_ids, target_class_id, frame_width, frame_height)

                        if best_id is not None:
                            tracked_subject_id = best_id
                            last_known_center = (best_center[0] + args.color_offset_x,
                                                 best_center[1] + args.color_offset_y)
                            subject_found_in_frame = True
                            if old_id is None:
                                print(
                                    f"Primary subject ({target_subject_name}) acquired with track ID: {tracked_subject_id}")
                            else:
                                print(
                                    f"Subject lost. Re-acquired new best target. Old ID: {old_id}, New ID: {tracked_subject_id}")

            # --- Frame Cropping and Centering Logic ---
            target_center = last_crop_center  # Default to last position
            if subject_found_in_frame and last_known_center is not None:
                center_history.append(last_known_center)
                smooth_center = np.mean(center_history, axis=0, dtype=int)
                # (Smoothing logic...)
                if len(center_history) > 1:
                    prev_smooth_center = np.mean(
                        list(center_history)[:-1], axis=0, dtype=int)
                    if np.linalg.norm(smooth_center - prev_smooth_center) > args.max_pixel_shift:
                        direction = (smooth_center - prev_smooth_center) / \
                            np.linalg.norm(smooth_center - prev_smooth_center)
                        smooth_center = prev_smooth_center + direction * args.max_pixel_shift
                target_center = smooth_center.astype(int)

            # --- Cropping and Padding Logic ---
            # Calculate the ideal crop coordinates
            ideal_crop_x1 = target_center[0] - args.width // 2
            ideal_crop_y1 = target_center[1] - args.height // 2
            ideal_crop_x2 = ideal_crop_x1 + args.width
            ideal_crop_y2 = ideal_crop_y1 + args.height

            if args.allow_offscreen:
                # Create a black canvas of the output size
                output_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

                # Find the intersection of the ideal crop and the frame
                src_x1 = max(0, ideal_crop_x1)
                src_y1 = max(0, ideal_crop_y1)
                src_x2 = min(frame_width, ideal_crop_x2)
                src_y2 = min(frame_height, ideal_crop_y2)

                # Calculate where to place the cropped section on the black canvas
                dest_x1 = max(0, -ideal_crop_x1)
                dest_y1 = max(0, -ideal_crop_y1)
                dest_x2 = dest_x1 + (src_x2 - src_x1)
                dest_y2 = dest_y1 + (src_y2 - src_y1)

                # If there is a valid intersection, copy the frame data
                if src_x1 < src_x2 and src_y1 < src_y2:
                    output_frame[dest_y1:dest_y2, dest_x1:dest_x2] = frame[src_y1:src_y2, src_x1:src_x2]

                cropped_frame = output_frame
                # Update last_crop_coords for logging purposes
                last_crop_coords = {"x1": int(ideal_crop_x1), "y1": int(ideal_crop_y1), "x2": int(ideal_crop_x2), "y2": int(ideal_crop_y2)}

            else:
                # Original logic: Clamp the crop box to the frame boundaries
                crop_x1 = max(0, min(ideal_crop_x1, frame_width - args.width))
                crop_y1 = max(0, min(ideal_crop_y1, frame_height - args.height))
                crop_x2 = crop_x1 + args.width
                crop_y2 = crop_y1 + args.height

                last_crop_coords = {"x1": int(crop_x1), "y1": int(crop_y1), "x2": int(crop_x2), "y2": int(crop_y2)}
                cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            current_crop_center = (
                last_crop_coords["x1"] + args.width // 2, last_crop_coords["y1"] + args.height // 2)
            delta_x, delta_y = current_crop_center[0] - \
                last_crop_center[0], current_crop_center[1] - \
                last_crop_center[1]
            print(
                f"Processing frame {frame_count}/{total_frames} | Shift (X, Y): ({delta_x}, {delta_y})", flush=True)
            last_crop_center = current_crop_center

            if args.visualize:
                # Draw the crop box (cyan), the detected subject center (red dot),
                # and — in color-tracking mode — the color blob's bounding box (orange).
                vis_frame = frame.copy()
                cv2.rectangle(
                    vis_frame,
                    (last_crop_coords["x1"], last_crop_coords["y1"]),
                    (last_crop_coords["x2"], last_crop_coords["y2"]),
                    (255, 255, 0), 6,
                )
                if args.track_color and last_known_color_bbox is not None:
                    bx, by, bw, bh = last_known_color_bbox
                    cv2.rectangle(vis_frame, (bx, by), (bx + bw, by + bh),
                                  (0, 165, 255), 4)
                if subject_found_in_frame and last_known_center is not None:
                    cv2.circle(vis_frame, last_known_center, 30, (0, 0, 255), -1)
                vis_out = cv2.resize(vis_frame, (pipe_w, pipe_h))
                ffmpeg_process.stdin.buffer.write(vis_out.tobytes())
            else:
                if cropped_frame.shape[1] != args.width or cropped_frame.shape[0] != args.height:
                    cropped_frame = cv2.resize(
                        cropped_frame, (args.width, args.height))
                ffmpeg_process.stdin.buffer.write(cropped_frame.tobytes())

    except BrokenPipeError:
        print("[Python] FFmpeg process pipe broke. This usually means FFmpeg closed prematurely.", flush=True)
    except Exception as e:
        print(f"[Python] An unexpected error occurred: {e}", flush=True)
    finally:
        cap.release()
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        err_thread.join()
        if ffmpeg_process.returncode != 0:
            print(
                f"[Python] FFmpeg exited with a non-zero status code: {ffmpeg_process.returncode}", flush=True)
        else:
            print(
                f"\nVideo processing complete. Output saved to {args.output_video}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Stabilize video by tracking a subject, preserving audio and quality.")
    parser.add_argument('input_video', type=str,
                        help="Path to the input video file.")
    parser.add_argument('output_video', type=str,
                        help="Path to save the stabilized output video file.")
    parser.add_argument('--target_subject', type=str, default='person',
                        help="The class of the subject to track (e.g., 'person', 'car', 'dog').")
    parser.add_argument('--model', type=str, default='yolov8l.pt',
                        help="YOLOv8 model to use (e.g., yolov8n.pt, yolov8l.pt).")
    parser.add_argument('--width', type=int, default=1536,
                        help="Width of the output video.")
    parser.add_argument('--height', type=int, default=1536,
                        help="Height of the output video.")
    parser.add_argument('--max_pixel_shift', type=int, default=50,
                        help="Maximum pixel shift for motion control.")
    parser.add_argument('--smoothing_window', type=int, default=10,
                        help="Number of frames to average for smoothing.")
    parser.add_argument('--device', type=str, default=None,
                        help="Compute device: cuda / mps / cpu. Auto-detected if omitted.")
    parser.add_argument('--video_codec', type=str, default=None,
                        help="FFmpeg video codec (e.g., 'hevc_nvenc' GPU NVIDIA, 'hevc_videotoolbox' GPU Apple, 'libx265' CPU). Auto-selected from --device if omitted.")
    parser.add_argument('--crf', type=int, default=16,
                        help="Constant Rate Factor for quality (lower is better, 10 is high quality for H.265/HEVC). Ignored when --bitrate is set.")
    parser.add_argument('--bitrate', type=str, default=None,
                        help="Target video bitrate (e.g. '50M'). Overrides --crf. Useful for matching source bitrate.")
    parser.add_argument('--conf', type=float, default=0.4,
                        help="Detection confidence threshold for the tracker.")
    parser.add_argument('--allow_offscreen', action='store_true',
                        help="Allow the crop box to go offscreen, creating black bars.")
    parser.add_argument('--visualize', action='store_true',
                        help="Output a visualization of the source video with the crop box and detected subject point overlaid, instead of the cropped video. Output is downscaled to 1920x1080 for easy preview.")
    parser.add_argument('--track_csv', type=str, default=None,
                        help="Path to a per-frame track CSV (header: frame,x,y,confidence). "
                        "Skips YOLO/color and uses (x,y) per frame as the centroid. "
                        "Generate from a DLC SuperAnimal-Bird H5 with dlc_to_track.py.")
    parser.add_argument('--track_color', action='store_true',
                        help="Use HSV color thresholding instead of YOLO. Requires at least one --color_range. Faster + more stable when the subject has a distinctive color marker (e.g. red-wing blackbird epaulet).")
    parser.add_argument('--color_range', type=parse_hsv_range, action='append',
                        default=[],
                        help="HSV range to threshold for, format 'H,S,V:H,S,V'. Repeatable; multiple ranges are OR'd together (use two ranges for hue-wraparound colors like red, e.g. --color_range 0,140,90:12,255,255 --color_range 168,140,90:180,255,255).")
    parser.add_argument('--color_min_area', type=int, default=200,
                        help="Ignore color blobs smaller than this many pixels (default: 200).")
    parser.add_argument('--color_max_area', type=int, default=80000,
                        help="Ignore color blobs larger than this many pixels (default: 80000).")
    parser.add_argument('--color_offset_x', type=int, default=0,
                        help="Pixel offset added to the detected centroid's X (positive = right). Applies to BOTH color and YOLO tracking modes. Use when the detected point isn't where you want the crop center.")
    parser.add_argument('--color_offset_y', type=int, default=0,
                        help="Pixel offset added to the detected centroid's Y (positive = down). Applies to BOTH color and YOLO tracking modes. E.g. for a red-winged blackbird's shoulder patch use +200 to frame the body; for a vertical bird (woodpecker) use a negative value to bias toward the head.")
    parser.add_argument('--auto_crop', action='store_true',
                        help="First-pass probe the video to find the centroid range, then auto-compute the largest 16:9 crop that can follow the centroid (with offset applied) without ever clamping to the source edges. Overrides --width and --height. Color-tracking mode only.")
    parser.add_argument('--auto_crop_percentile', type=int, default=0,
                        help="When set (e.g. 2), --auto_crop uses the p<N>..p<100-N> range of centroid positions instead of strict min/max. Lets a few outlier detections clamp at edges instead of collapsing the whole crop. Recommended 1-3 for DLC tracks where misdetections are rare but possible.")

    args = parser.parse_args()
    if args.device is None:
        args.device = autodetect_device()
    if args.video_codec is None:
        args.video_codec = autodetect_codec(args.device)
    main(args)
