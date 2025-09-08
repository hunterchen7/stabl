import cv2
import argparse
import numpy as np
import subprocess
import threading


def reader_thread(pipe, stream_name):
    """A simple thread function to read from a subprocess pipe and print."""
    try:
        for line in iter(pipe.readline, ''):
            print(f"[{stream_name}] {line.strip()}", flush=True)
    finally:
        pipe.close()


def main(args):
    """
    Main function to draw the crop box from a stabilized video onto the original video.
    """
    print("Opening video files...")
    # Open the original video to get its properties and frames
    cap_orig = cv2.VideoCapture(args.original_video)
    if not cap_orig.isOpened():
        print(
            f"Error: Could not open original video file: {args.original_video}")
        return

    # Open the stabilized video to use as the 'template'
    cap_stab = cv2.VideoCapture(args.stabilized_video)
    if not cap_stab.isOpened():
        print(
            f"Error: Could not open stabilized video file: {args.stabilized_video}")
        cap_orig.release()
        return

    # Get properties from the ORIGINAL video for the output
    frame_width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"Original video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

    # --- FFmpeg Subprocess for High-Quality Output with Audio ---
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists

        # === Input Pipe Settings (from Python script) ===
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        # DYNAMIC: Must match the original video's resolution
        '-s', f'{frame_width}x{frame_height}',
        # DYNAMIC: Must match the original video's framerate
        '-r', str(fps),
        # Tells FFmpeg to read video frames from stdin (the pipe)
        '-i', '-',

        # === Input File Settings (for audio) ===
        '-i', args.original_video,            # DYNAMIC: The original video file for audio

        # === HARD-CODED Video Encoding Settings ===
        # HARD-CODED: Use the NVIDIA HEVC (H.265) encoder
        '-c:v', 'hevc_nvenc',
        # HARD-CODED: Downscale the output to 1920x1080
        '-vf', 'scale=1920:1080',
        # HARD-CODED: 'p5' is a good balance of speed and quality
        '-preset', 'p5',
        # HARD-CODED: '23' is a good constant quality level
        '-cq', '23',

        # === Audio and Stream Mapping Settings ===
        '-c:a', 'copy',                       # Copy the audio stream without re-encoding
        '-map', '0:v:0',                      # Map the video from the pipe to the output
        '-map', '1:a:0?',                     # Map the audio from the file to the output

        # === Output File ===
        args.output_video,                    # DYNAMIC: The output file path
    ]

    ffmpeg_process = subprocess.Popen(
        ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    err_thread = threading.Thread(target=reader_thread, args=(
        ffmpeg_process.stderr, "ffmpeg_stderr"), daemon=True)
    err_thread.start()
    print("\nFFmpeg process started for output. Beginning frame analysis...\n")

    frame_count = 0
    try:
        while True:
            # Read one frame from each video
            ret_orig, frame_orig = cap_orig.read()
            ret_stab, frame_stab = cap_stab.read()

            # If either video has ended, stop the loop
            if not ret_orig or not ret_stab:
                break

            frame_count += 1
            print(
                f"Processing frame {frame_count}/{total_frames}...", flush=True)

            # --- Template Matching ---
            # Find the location of the stabilized frame (template) within the original frame
            # TM_CCOEFF_NORMED is a reliable matching method
            result = cv2.matchTemplate(
                frame_orig, frame_stab, cv2.TM_CCOEFF_NORMED)

            # Get the coordinates of the best match
            _minVal, _maxVal, _minLoc, maxLoc = cv2.minMaxLoc(result)
            top_left = maxLoc

            # Get the dimensions of the stabilized frame to calculate the bottom right corner
            h, w = frame_stab.shape[:2]
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # --- Draw the Rectangle ---
            # Draw a red rectangle on the original frame at the matched location
            # Color is BGR format: (Blue, Green, Red)
            # Thickness is in pixels
            cv2.rectangle(frame_orig, top_left, bottom_right,
                          (0, 0, 255), args.thickness)

            # Write the modified original frame to the FFmpeg process
            ffmpeg_process.stdin.buffer.write(frame_orig.tobytes())

    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        # --- Clean Up ---
        print("\nFinalizing video output...")
        cap_orig.release()
        cap_stab.release()
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        err_thread.join()

        if ffmpeg_process.returncode == 0:
            print(
                f"Successfully created visualization video: {args.output_video}")
        else:
            print(
                f"FFmpeg failed with return code {ffmpeg_process.returncode}. Please check the [ffmpeg_stderr] messages above.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Draws the crop box from a stabilized video onto the original video for visualization.")
    parser.add_argument('original_video', type=str,
                        help="Path to the original, high-resolution video file.")
    parser.add_argument('stabilized_video', type=str,
                        help="Path to the stabilized, cropped video file.")
    parser.add_argument('output_video', type=str,
                        help="Path to save the final visualization video.")
    parser.add_argument('--thickness', type=int, default=4,
                        help="The thickness of the red rectangle border in pixels.")

    args = parser.parse_args()
    main(args)
