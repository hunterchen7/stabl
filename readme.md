This is a Python program that stabilizes video footage by tracking a subject and cropping the frame to keep it centered. This program utilizes the powerful YOLOv8 library for object detection and tracking, OpenCV for video manipulation, and FFmpeg for robust video encoding. It's designed to be run from the command line with various options for customization.

### Features

- **Subject Tracking:** Uses a pre-trained YOLOv8 model to detect and track objects in the video. It identifies the most prominent subject and follows it across frames.
- **Cropped Stabilization:** Creates a stabilized video by keeping the tracked subject in the center of a cropped frame.
- **Customizable Output:** Allows you to specify the dimensions of the output video.
- **Motion Smoothing:** Implements a simple smoothing algorithm to prevent jerky movements of the cropping window.
- **GPU Acceleration:** Leverages your NVIDIA GPU for both object detection and video encoding/decoding to significantly speed up the process.
- **Command-Line Interface:** Provides a user-friendly command-line interface to specify input/output files and other parameters.

### Prerequisites

Before running the program, you need to have the following installed:

1.  **Python 3.8 or later**
2.  **FFmpeg:** It needs to be installed on your system and accessible from the command line. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).
3.  **NVIDIA GPU with CUDA support:** For GPU acceleration, you'll need a compatible NVIDIA graphics card with the necessary CUDA drivers installed.
4.  **Python Libraries:** You can install the required Python packages using pip:

```bash
pip install opencv-python ultralytics numpy
```

### How to Run the Program

1.  Save the code above as a Python file named `stabilize.py`.
2.  Open your console or terminal.
3.  Navigate to the directory where you saved `stabilize.py`.
4.  Run the program with the required arguments.

**Basic Usage:**

```bash
python stabl.py "path/to/your/input_video.mp4" "path/to/your/stabilized_video.mp4"
```

**Example with Optional Arguments:**

This example will create a 1536x1536 stabilized video with a maximum frame-to-frame shift of 400 pixels, tracking an airplane as the target subject, and using a smoothing window of 3 frames.

```bash
python stabl.py "examples/C0018.MP4" "examples/C0018_stabilized.mp4" --width 1536 --height 1536 --target_subject airplane --smoothing_window 3 --max_pixel_shift 400
```
