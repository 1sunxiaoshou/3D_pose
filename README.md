# 3D Pose Estimation Project

This project consists of several Python scripts that work together to estimate 3D human poses from images or videos using deep learning models and computer vision techniques.

## Files Overview

### 1. 3D_Pose.py
- **Description**: Estimates 3D human poses from images using an ONNX model.
- **Key Features**:
  - ONNX Runtime and OpenCV for image processing.
  - 3D joint position prediction.
  - 3D plotting of joint positions.

### 2. 骨骼点提取.py
- **Description**: Extracts human pose keypoints from images or videos using OpenPose.
- **Key Features**:
  - Configuration and initialization of OpenPose.
  - Processing of images and videos.
  - Visualization and optional saving of processed videos.

### 3. Pose_to_3D.py
- **Description**: A class-based approach to load a model, preprocess images, perform inference, and plot 3D keypoints.
- **Key Methods**:
  - `load_model`: Loads the ONNX model.
  - `preprocess_image`: Preprocesses the image for inference.
  - `inference_image`: Performs inference to extract joint positions.
  - `plot_keypoints_3d`: Plots the 3D joint positions.

### 4. main.py
- **Description**: A PyQt5-based GUI application for loading images or videos, processing them through the 3D pose estimation class, and displaying the results.
- **Key Components**:
  - `PoseEstimator_Child`: Inherits from `PoseEstimator` in `Pose_to_3D.py`, adds 2D plotting capabilities.
  - `MyMainWindow`: Main window class with buttons and event handling logic for file selection, processing, and result display.

## Usage
- Run `main.py` to start the GUI application.
- Load an image or video file and process it to view the 3D pose estimation results.

## Dependencies
- Python 3.x
- ONNX Runtime
- OpenCV
- OpenPose
- PyQt5
- TensorFlow (for certain functionalities)

## License
This project is open-source and available under the MIT License.
