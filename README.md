# StereoXtrinsicsUI

Interactive GUI for dual-camera extrinsic calibration (estimate relative rotation R and translation t between two cameras). Designed for quick experiment and prototyping.

> TL;DR: Load stereo image pairs, detect a target chessboard, refine, visualize camera poses, and export result R|t (YAML).

## Installation

```bash
git clone https://github.com/yunjinli/StereoXtrinsicsUI.git
cd StereoXtrinsicsUI
conda create -n StereoXtrinsicsUI python=3.9
conda activate StereoXtrinsicsUI
pip install -r requirements.txt
```

## Run

### Intrinsics

The UI assume the intrinsics for both cameras are known. You can use some tools like [this](https://docs.ros.org/en/kilted/p/camera_calibration/doc/tutorial_stereo.html) to get their intrinsics first.

### Data Preparation

Please save image pairs to `</path/to/imagepairs>/cam0/` and `</path/to/imagepairs>/cam1/` and make sure that the pairs are synchronized. You can also use my ROS2 [package](https://github.com/yunjinli/data_pipeline) to do so.

### Prepare Configuration

```yaml
img_path: /home/yunjinli/camera_calibration/rel_calib_images/2 ## base path contraining cam0/* and cam1/*
chessboard_size: [8, 6]
square_size: 0.06
cam_dict: { 0: "realsense", 1: "tof" } ## For name display
undistorted: False
K0:
  [
    [917.626363, 0.000000, 631.016065],
    [0.000000, 920.528195, 345.491278],
    [0.000000, 0.000000, 1.000000],
  ]
D0: [0.082749, -0.124803, -0.001593, -0.000622, 0.000000]
h0: 720
w0: 1280
K1:
  [
    [205.731476, 0.000000, 112.496540],
    [0.000000, 206.057320, 85.766603],
    [0.000000, 0.000000, 1.000000],
  ]
D1: [0.287139, -0.740035, 0.000383, -0.001975, 0.000000]
h1: 172
w1: 224
```

### Run

```bash
python UI.py -W 1920 -H 1080 --config ./config/default.yaml
```

## Docker

### Step1: Build the image
```
docker build -f docker/Dockerfile -t stereox-ui:latest .
```
### Step2: Modify your default.yaml
### Step3: (MacOS) Launch the app
if you have display issue, this is how I solve it:
- Step1: you have to properly set XQuartz → Preferences → Security to “Allow connections from network clients”
- Step2: Enable indirect GLX (needed for OpenGL/GLUT over the network)
```
defaults write org.xquartz.X11 enable_iglx -bool true
```
- Step3: Quit and relaunch XQuartz
- Step4: Allow connections from Docker’s VM subnet (Docker Desktop usually uses 192.168.65.0/24) and localhost:
```
xhost +localhost
xhost + 192.168.65.0/24
```
Launch the container:
```
## Make sure to pass the correct host path to be mounted to the container

CFG=./config/default.yaml IMG_DIR=~/Downloads/4/ docker compose up stereox-macwin
```

## Demo

Demo on the UI. Note that this example takes quite unusual camera mounting setup (90 degree offset and almost 7cm offset for the baseline). By using the UI, we can still get the correct extrisics easily.

https://github.com/user-attachments/assets/3c807531-1c99-4d29-b73a-3dcdb802af7b

## Results in Rviz

By using the calibrated extrinsic to colorized the point cloud produced by PMD Flexx2 ToF Camera with RGB colors from Intel Realsense D415.

https://github.com/user-attachments/assets/023702e4-18d4-4df7-a245-437849693df9
