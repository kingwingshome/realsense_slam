# RealSense D455 Visual SLAM

A simple Visual SLAM (Simultaneous Localization and Mapping) system using Intel RealSense D455 RGB-D camera. This project implements feature-based SLAM with ORB features and RGB-D depth information.

## Features

- **RGB-D SLAM**: Utilizes both color and depth data from RealSense D455
- **ORB Features**: Fast and efficient feature extraction using ORB
- **Real-time Tracking**: Live camera pose estimation and mapping
- **Keyframe Management**: Automatic keyframe selection for efficient mapping
- **Trajectory Recording**: Save camera trajectory for analysis
- **Visualization**: Real-time visualization of features, tracking status, and camera pose
- **Trajectory Replayer**: Visualize and replay recorded trajectory data with animation

## Hardware Requirements

- Intel RealSense D455 camera
- USB 3.0 or higher connection
- Computer with sufficient processing power (recommended: Intel i5 or equivalent)

## Software Requirements

- Python 3.8 or higher
- Intel RealSense SDK 2.0
- OpenCV 4.8 or higher

## Installation

### 1. Install Intel RealSense SDK

**Windows:**
```bash
# Download and install from Intel website
# https://www.intelrealsense.com/sdk-2/
```

**Linux (Ubuntu):**
```bash
# Register server's public key
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo apt-add-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"

# Install library
sudo apt-get update
sudo apt-get install librealsense2-utils librealsense2-dev librealsense2-dkms

# Install Python wrapper
sudo apt-get install python3-realsense2
```

**macOS:**
```bash
# Install via Homebrew
brew install librealsense
```

### 2. Install Python Dependencies

```bash
cd realsense_slam
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pyrealsense2 opencv-python opencv-contrib-python numpy pyyaml matplotlib colorlog
```

### 3. Verify Installation

Test the camera connection:
```bash
python camera_capture.py
```

You should see RGB and depth video streams from your D455 camera.

## Usage

### Basic Usage

Run the SLAM system with default settings:
```bash
python main.py
```

### Command Line Options

```bash
python main.py --width 640 --height 480 --fps 30 --output-dir output
```

Options:
- `--width`: Image width (default: 640)
- `--height`: Image height (default: 480)
- `--fps`: Frames per second (default: 30)
- `--output-dir`: Output directory for results (default: output)

### Controls

While running, use the following keyboard controls:

| Key | Action |
|-----|--------|
| `q` | Quit and save results |
| `s` | Save trajectory to file |
| `k` | Save current frame as keyframe |
| `r` | Reset SLAM system |
| `d` | Toggle depth display |

### Configuration

Edit [`config.yaml`](config.yaml) to customize SLAM parameters:

```yaml
camera:
  width: 640
  height: 480
  fps: 30

slam:
  orb:
    nfeatures: 2000
    scale_factor: 1.2
    nlevels: 8
  matching:
    ratio_threshold: 0.7
    min_matches: 10
```

## Project Structure

```
realsense_slam/
├── camera_capture.py         # RealSense D455 camera wrapper
├── visual_slam.py            # Visual SLAM implementation
├── main.py                   # Main entry point
├── trajectory_replayer.py    # Trajectory visualization and replayer
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Output Files

After running the SLAM system, the following files are generated in the output directory:

- `trajectory.txt`: Camera trajectory (x, y, z positions)
- `trajectory_with_timestamps.txt`: Trajectory with timestamps
- `keyframe_XXXXXX.png`: Saved keyframes (if saved manually)

## How It Works

### 1. Camera Capture

The [`RealSenseD455`](camera_capture.py:16) class handles:
- Initializing the RealSense pipeline
- Capturing aligned RGB and depth frames
- Providing camera intrinsics

### 2. Feature Extraction

The [`VisualSLAM`](visual_slam.py:18) class implements:
- ORB feature detection and description
- Feature matching using Lowe's ratio test
- Pose estimation from essential matrix

### 3. RGB-D SLAM

- Depth information is used to directly compute 3D point positions
- Camera pose is estimated from matched features
- Keyframes are selected based on feature overlap

## Troubleshooting

### Camera Not Detected

```bash
# Check if camera is connected
realsense-viewer  # Linux
rs-enumerate-devices  # Windows/Linux
```

### Import Errors

```bash
# Reinstall pyrealsense2
pip uninstall pyrealsense2
pip install pyrealsense2
```

### Low Frame Rate

- Reduce image resolution: `python main.py --width 320 --height 240`
- Reduce FPS: `python main.py --fps 15`
- Reduce number of features in [`config.yaml`](config.yaml:11)

### Tracking Loss

- Ensure sufficient lighting
- Move camera slowly
- Increase number of features in configuration

## Advanced Usage

### Using Individual Modules

```python
from camera_capture import RealSenseD455
from visual_slam import VisualSLAM

# Initialize camera
camera = RealSenseD455(width=640, height=480, fps=30)
camera.initialize()

# Get intrinsics and initialize SLAM
intrinsics = camera.get_camera_intrinsics()
slam = VisualSLAM(intrinsics)

# Process frames
while True:
    color, depth, _ = camera.get_frame()
    result = slam.process_frame(color, depth)
    print(f"Pose: {result['pose'][:3, 3]}")
```

### Visualizing Trajectory

```python
import numpy as np
import matplotlib.pyplot as plt

# Load trajectory
trajectory = np.loadtxt('output/trajectory.txt')

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.title('Camera Trajectory')
plt.show()
```

## Trajectory Replayer

The [`trajectory_replayer.py`](trajectory_replayer.py) module provides tools to visualize and replay recorded SLAM trajectory data.

### Features

- **3D Visualization**: View trajectory in 3D with start/end markers
- **2D Projections**: View trajectory from top, side, and front perspectives
- **Animation**: Replay trajectory with customizable speed
- **Statistics**: Display trajectory statistics (path length, duration, bounding box)
- **Export**: Export to JSON, KITTI, or TUM formats

### Basic Usage

#### Static 3D Plot

```bash
python trajectory_replayer.py output/trajectory.txt
```

#### 2D Projections

```bash
python trajectory_replayer.py output/trajectory.txt --2d
```

#### Animated Playback

```bash
python trajectory_replayer.py output/trajectory.txt --animate
```

#### With Timestamps

```bash
python trajectory_replayer.py output/trajectory.txt --timestamps output/trajectory_with_timestamps.txt
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `trajectory_file` | Path to trajectory file (required) |
| `--timestamps` | Path to timestamps file |
| `--animate` | Animate trajectory playback |
| `--2d` | Show 2D projections |
| `--interval` | Animation interval in milliseconds (default: 50) |
| `--save-plot` | Path to save static plot |
| `--save-animation` | Path to save animation (GIF format) |
| `--export-json` | Export trajectory to JSON format |
| `--export-kitti` | Export trajectory to KITTI format |
| `--export-tum` | Export trajectory to TUM format |

### Examples

**Save static plot:**
```bash
python trajectory_replayer.py output/trajectory.txt --save-plot trajectory_plot.png
```

**Save animation:**
```bash
python trajectory_replayer.py output/trajectory.txt --animate --save-animation trajectory.gif
```

**Export to JSON:**
```bash
python trajectory_replayer.py output/trajectory.txt --export-json trajectory.json
```

**Export to KITTI format:**
```bash
python trajectory_replayer.py output/trajectory.txt --export-kitti trajectory_kitti.txt
```

**Slow animation (100ms interval):**
```bash
python trajectory_replayer.py output/trajectory.txt --animate --interval 100
```

### Using as a Python Module

```python
from trajectory_replayer import TrajectoryReplayer

# Load trajectory
replayer = TrajectoryReplayer('output/trajectory.txt', 'output/trajectory_with_timestamps.txt')

# Print statistics
replayer.print_statistics()

# Show static plot
replayer.plot_trajectory()

# Show 2D projections
replayer.plot_2d_projections()

# Animate trajectory
replayer.animate_trajectory(interval=50)

# Export to different formats
replayer.export_to_json('trajectory.json')
replayer.export_to_kitti('trajectory_kitti.txt')
replayer.export_to_tum('trajectory_tum.txt')
```

### Output Formats

#### JSON Format
```json
{
  "trajectory": [[x1, y1, z1], [x2, y2, z2], ...],
  "timestamps": [t1, t2, ...],
  "metadata": {
    "num_points": 1000,
    "duration": 33.33,
    "path_length": 12.45
  }
}
```

#### KITTI Format
```
timestamp tx ty tz qx qy qz qw
```

#### TUM Format
```
timestamp tx ty tz qx qy qz qw
```

## References

- [Intel RealSense Documentation](https://dev.intelrealsense.com/docs)
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [OpenCV Documentation](https://docs.opencv.org/)

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Intel RealSense team for the excellent SDK
- OpenCV community for the computer vision tools
- ORB-SLAM3 team for SLAM algorithms inspiration
