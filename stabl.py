import cv2
import argparse
import numpy as np
import subprocess
import threading
from ultralytics import YOLO
from collections import deque


def reader_thread(pipe, stream_name):
    """A simple thread function to read from a subprocess pipe and print."""
    try:
        # Read line by line in text mode
        for line in iter(pipe.readline, ''):
            print(f"[{stream_name}] {line.strip()}", flush=True)
    finally:
        pipe.close()


def main(args):
    """
    Main function to process the video stabilization with audio and quality preservation.
    """
    # ... (YOLO model loading and subject validation is the same)
    # Load the specified YOLOv8 model with GPU support
    print(f"Loading YOLO model: {args.model}...")
    try:
        model = YOLO(args.model)
        model.to('cuda')
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

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- FFmpeg Subprocess Setup ---
    ffmpeg_command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{args.width}x{args.height}',
        '-r', str(fps), '-i', '-', '-i', args.input_video,
        '-c:v', args.video_codec, '-preset', 'slow', '-crf', str(args.crf),
        '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0?',
        args.output_video,
    ]

    # --- MODIFIED SUBPROCESS HANDLING ---
    # We now capture stderr and use text mode for the reader thread
    ffmpeg_process = subprocess.Popen(
        ffmpeg_command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # Use text mode for stderr to get strings
    )

    # Create and start the thread to read FFmpeg's stderr
    # The 'daemon=True' ensures the thread will exit when the main program does
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

            # --- Object Detection and Tracking ---
            results = model.track(frame, persist=True,
                                  device='cuda', verbose=False)
            subject_found_in_frame = False
            # (The full, correct logic from the previous step goes here)
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes, track_ids, classes = results[0].boxes.xyxy.cpu().numpy(
                ), results[0].boxes.id.int().cpu().tolist(), results[0].boxes.cls.int().cpu().tolist()
                if tracked_subject_id is None:
                    best_candidate_id, max_score = None, 0
                    center_of_frame = np.array(
                        [frame_width / 2, frame_height / 2])
                    for i, box in enumerate(boxes):
                        if classes[i] == target_class_id:
                            score = (box[2] - box[0]) * (box[3] - box[1]) / (np.linalg.norm(np.array(
                                [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) - center_of_frame) + 1e-6)
                            if score > max_score:
                                max_score, best_candidate_id = score, track_ids[i]
                    if best_candidate_id is not None:
                        tracked_subject_id = best_candidate_id
                        print(
                            f"Primary subject ({target_subject_name}) identified with track ID: {tracked_subject_id}")

                if tracked_subject_id in track_ids:
                    subject_index = track_ids.index(tracked_subject_id)
                    if classes[subject_index] == target_class_id:
                        subject_found_in_frame = True
                        x1, y1, x2, y2 = boxes[subject_index]
                        last_known_center = (
                            int((x1 + x2) / 2), int((y1 + y2) / 2))

            # --- Frame Cropping and Centering Logic ---
            target_center = last_crop_center
            if subject_found_in_frame and last_known_center is not None:
                center_history.append(last_known_center)
                smooth_center = np.mean(center_history, axis=0, dtype=int)
                if len(center_history) > 1:
                    prev_smooth_center = np.mean(
                        list(center_history)[:-1], axis=0, dtype=int)
                    if np.linalg.norm(smooth_center - prev_smooth_center) > args.max_pixel_shift:
                        direction = (smooth_center - prev_smooth_center) / \
                            np.linalg.norm(smooth_center - prev_smooth_center)
                        smooth_center = prev_smooth_center + direction * args.max_pixel_shift
                target_center = smooth_center.astype(int)

            crop_x1 = target_center[0] - args.width // 2
            crop_y1 = target_center[1] - args.height // 2
            crop_x1 = max(0, min(crop_x1, frame_width - args.width))
            crop_y1 = max(0, min(crop_y1, frame_height - args.height))
            last_crop_coords = {"x1": int(crop_x1), "y1": int(crop_y1), "x2": int(
                crop_x1 + args.width), "y2": int(crop_y1 + args.height)}

            current_crop_center = (
                last_crop_coords["x1"] + args.width // 2, last_crop_coords["y1"] + args.height // 2)
            delta_x, delta_y = current_crop_center[0] - \
                last_crop_center[0], current_crop_center[1] - \
                last_crop_center[1]
            print(
                f"Processing frame {frame_count}/{total_frames} | Shift (X, Y): ({delta_x}, {delta_y})", flush=True)
            last_crop_center = current_crop_center

            cropped_frame = frame[last_crop_coords["y1"]:last_crop_coords["y2"],
                                  last_crop_coords["x1"]:last_crop_coords["x2"]]
            if cropped_frame.shape[1] != args.width or cropped_frame.shape[0] != args.height:
                cropped_frame = cv2.resize(
                    cropped_frame, (args.width, args.height))

            # --- Write Frame to FFmpeg's stdin (raw bytes) ---
            ffmpeg_process.stdin.buffer.write(cropped_frame.tobytes())
    except BrokenPipeError:
        print("[Python] FFmpeg process pipe broke. This usually means FFmpeg closed prematurely.", flush=True)
    except Exception as e:
        print(f"[Python] An unexpected error occurred: {e}", flush=True)
    finally:
        # --- Finalize ---
        cap.release()
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()  # Close stdin to signal EOF to FFmpeg

        ffmpeg_process.wait()  # Wait for FFmpeg to finish
        err_thread.join()  # Wait for the reader thread to finish

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
    parser.add_argument('--video_codec', type=str, default='libx264',
                        help="FFmpeg video codec (e.g., 'libx264', 'h264_nvenc' for GPU).")
    parser.add_argument('--crf', type=int, default=18,
                        help="Constant Rate Factor for quality (lower is better, 18 is ~visually lossless).")

    args = parser.parse_args()
    main(args)
