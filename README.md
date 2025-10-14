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

```bash
python UI.py -W 1920 -H 1080 --config ./config/default.yaml
```

## Configuration

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

## Demo

https://github.com/user-attachments/assets/3c807531-1c99-4d29-b73a-3dcdb802af7b

## Results in Rviz

https://github.com/user-attachments/assets/023702e4-18d4-4df7-a245-437849693df9
