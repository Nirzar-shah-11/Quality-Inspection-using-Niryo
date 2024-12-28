

# Bottle Inspection and Classification with Niryo Robot

This project uses the Niryo robot arm to detect, classify, and sort bottles as defective or non-defective. The system integrates **YOLOv8** and **MobileNet SSD** for object detection, along with various approaches for handling bottle detection and sorting.

## Features
- **Object Detection**: Identifies bottles in the camera feed using **YOLOv8** and **MobileNet SSD**.
- **Defect Classification**: Uses MobileNet SSD for defect detection and sorting.
- **Robotic Sorting**: Uses the Niryo robot arm to pick and place bottles based on detection.
- **Robot Calibration**: Ensures the robot is calibrated for accurate positioning.

---

## File Structure

### Directories

#### YOLOv8
- **yolo8/CremeBottleDetection**
  - `yolov8.yaml`: Configuration file for YOLOv8 basic detection.
  - `yolov8.pt`: Pre-trained weights for YOLOv8 basic detection.

- **VisionPickAndPlace**
  - `approach1.py`: Python file for basic detection and pick-and-place.
  - `approach2.py`: Python file for vision pick-and-place.
  - `approach3.py`: Python file for 2-camera approach.

#### MobileNet SSD + CNN
- **mobilenet+CNN/CNN**
  - `deploy.prototxt`: Configuration file for MobileNet SSD.
  - `mobilenet_iter_73000.caffemodel`: Pre-trained weights for MobileNet SSD.
  - `bottle_defect_classifier.h5`: CNN model for defect classification.

- **training.py**: Python file for model training.
- **pickplacemobilenet.py**: Basic pick-and-place functionality.
- **approach1.py**: Basic bottle detection.
- **approach2.py**: Horizontal and vertical bottle detection.
- **calibrate.py**: Script for robot calibration.



---

## Requirements

### Hardware
- Niryo One robot with camera.
- Computer with Python installed.
- A workspace set up for sorting defective and non-defective bottles.

### Software and Libraries
- Python 3.x
- OpenCV
- TensorFlow/Keras
- Pyniryo
- NumPy
- YOLOv8 files (`yolov8.yaml` and `yolov8.pt`)
- MobileNet SSD files (`deploy.prototxt` and `mobilenet_iter_73000.caffemodel`)

---

## Setup

### Robot Configuration
1. Connect the Niryo robot to your network.
2. Use the IP address of the robot in the scripts (e.g., `10.10.10.10`).
3. Ensure the Niryo robot is calibrated using `calibrate.py`.

### Install Required Libraries
```bash
pip install opencv-python-headless tensorflow pyniryo numpy
```

### Place Files
1. Place the YOLOv8 files (`yolov8.yaml`, `yolov8.pt`) in the `yolo8/CremeBottleDetection` directory.
2. Place the MobileNet SSD files (`deploy.prototxt`, `mobilenet_iter_73000.caffemodel`) in the `mobilenet+CNN/CNN` directory.
3. Place the defect classification model (`bottle_defect_classifier.h5`) in the same directory.
4. Ensure dataset images are properly organized under `dataset/`.

---

## Running the Project

### YOLOv8 Object Detection Approaches

1. **Basic Detection**:
   ```bash
   python yolo8/CremeBottleDetection.py
   ```

2. **Vision Pick and Place**:
   ```bash
   python VisionPickAndPlace/approach1.py
   python VisionPickAndPlace/approach2.py
   python VisionPickAndPlace/approach3.py
   ```

### MobileNet SSD Object Detection Approaches

1. **Basic Pick-and-Place**:
   ```bash
   python pickplacemobilenet.py
   ```

2. **Bottle Detection**:
   ```bash
   python approach1.py
   ```

3. **Horizontal and Vertical Detection**:
   ```bash
   python approach2.py
   ```

### Robot Calibration
Calibrate the robot using the following script:
```bash
python calibrate.py
```

---

## Dataset Structure

- **YOLOv8**:
  - `dataset/good/`: Contains images of non-defective bottles (Good).
  - `dataset/bad/`: Contains images of defective bottles (Bad).

- **MobileNet SSD**:
  - `dataset/defective/`: Contains images of defective bottles.
  - `dataset/non_defective/`: Contains images of non-defective bottles.

---

## Notes
- Ensure the workspace is properly defined for the robot in the code (e.g., `QualityInspection1` workspace).
- Update any positional values in the scripts to match your workspace setup.
- Use the `convert_data` function to handle positional conversions for the Niryo robot.

---

## Troubleshooting
- If the robot cannot detect objects, ensure the camera is correctly connected and the detection threshold is appropriate.
- If the robot's movements are inaccurate, recalibrate using `calibrate.py`.

---

## Future Improvements
- Integrate a more advanced detection model for better accuracy.
- Add real-time feedback for robot operations.
- Optimize the classification model for faster inference.

---

## Acknowledgments
- **YOLOv8**: Used for object detection.
- **MobileNet SSD**: Used for object detection.
- **TensorFlow/Keras**: For defect classification model (only used in MobileNet SSD section).
- **Pyniryo**: Python library for Niryo robot control.

---

## License
This project is licensed under the MIT License.
