# **Quality Inspection Using Niryo**

This project utilizes the Niryo robot arm to detect, classify, and sort objects (e.g., bottles) as defective or non-defective. It integrates object detection using YOLOv8 and MobileNet SSD, along with robot calibration and automated pick-and-place operations.

---

## **Features**

- **Object Detection:** Utilizes YOLOv8 and MobileNet SSD to detect objects in the workspace.
- **Defect Classification:** MobileNet SSD with CNN-based classification for sorting defective and non-defective objects.
- **Robotic Sorting:** Niryo robot arm performs automated pick-and-place based on defect detection.
- **Robot Calibration:** Ensures the robot arm is accurately calibrated for object positioning.
- **Data Generation:** Scripts to create and augment datasets for training object detection and classification models.

---

## **Repository Structure**

### **1. Basic**

Contains scripts for basic robot operation and coordinate fetching:

- `PyNiryo.py`: Python bindings for interacting with the Niryo robot.
- `With_Yolo`: YOLO integrated with basic robot operations.
- `basic_grabbing.py`: Script for simple grabbing operations.
- `getting_coordinates.py`: Fetches coordinates from the workspace.
- `robot_camera_with_yolo.py`: Integrates camera operations with YOLO detection.
- `robot_correct_image_extraction.py`: Extracts corrected images for training datasets.
- `steel_can_detection.py`: Detects and sorts steel cans.
- `video.py`: Demonstrates video capture functionality.

---

### **2. Can Detection Using YOLO**

Contains scripts for YOLO-based object detection:

- `VisionSet_yolo_can_detection.py`: Configures and sets up vision-based detection using YOLO.
- `yolo_can_detection.py`: Performs YOLO-based can detection.
- `yolo_can_detection_training.py`: Training script for YOLO-based object detection.
- `yolov8_custom.pt`: Fine Tuned YOLOv8 model for Fanta can detection.

---

### **3. Image Generation**

Scripts to create and augment datasets for object detection and classification:

- `creating_dataset.py`: Captures images from the robot's camera for dataset generation.
- `data_augmentation.py`: Augments captured images for a robust training dataset.
- `data_creation.py`: Script for controlled image creation for datasets.

---

### **4. MobileNet SSD**

Implements MobileNet SSD for defect detection and robotic sorting:

- `Approach1.py`: Basic defect detection.
- `Approach2_Trail.py` & `Approach2_final.py`: Horizontal and vertical defect detection approaches.
- `Laptoptraining.py` & `Training.py`: Training scripts for MobileNet SSD and CNN models.
- `bottle_defect_classifier.h5`: Pre-trained model for defect classification.
- `calibrate.py`: Script for robot calibration.
- `mobilenet_iter_73000.caffemodel`: Pre-trained weights for MobileNet SSD.
- `deploy.prototxt`: MobileNet SSD configuration file.

#### **Dataset Structure**

- `dataset/defective/`: Contains images of defective objects.
- `dataset/non-defective/`: Contains images of non-defective objects.

---

### **5. YOLOv8**

Implements YOLOv8 for advanced object detection and pick-and-place operations:

- `Approach1.py`, `Approach2.py`: Various YOLO-based object detection approaches.
- `Approach3.py`: YOLO based approach using 2 cameras 
- `Approach4_trail.py`: Updated script for horizontal and vertical object detection.
- `approach4_updated.py`: Prototype for a new approach.
- `config1.yaml`: YOLOv8 configuration file for training.
- `yolov8_custom1.pt`: Fine Tuned YOLOv8 model for cream bottle detection.

---

### **6. Cream Bottle Detection**

Scripts for detecting cream bottles:

- `cream_bottle_detection.py`: Detects cream bottles using a custom YOLOv8 model.
- `yolov8_custom1.pt`: Fine Tuned YOLOv8 model for cream bottle detection.

---

## **Setup**

### **Hardware**

- Niryo One robot with a camera.
- Computer with Python installed.
- Workspace for sorting defective and non-defective objects.

### **Software Requirements**

Install the required Python libraries:

```bash
pip install opencv-python-headless tensorflow pyniryo numpy
```

### **File Placement**

- Place YOLOv8 configuration (`config1.yaml`) and weights (`yolov8_custom1.pt`) in the `YOLOv8` directory.
- Place MobileNet SSD files (`deploy.prototxt`, `mobilenet_iter_73000.caffemodel`) in the `MobileNetSSD` directory.
- Place the defect classification model (`bottle_defect_classifier.h5`) in the same directory.
- Ensure dataset images are placed in `dataset/defective/` and `dataset/non-defective/`.

---

## **Usage**

### **1. Object Detection (YOLOv8)**

- Run the basic detection script:
  ```bash
  python YOLOv8/Approach2.py
  ```
- Run the horizontal and vertical detection script:
  ```bash
  python YOLOv8/Approach4_Trail.py
  ```

### **2. MobileNet SSD**

- Perform basic defect classification:
  ```bash
  python MobileNetSSD/Approach1.py
  ```
 - Perform horizontal & Vertical defect classification:
  ```bash
  python MobileNetSSD/Approach2_updated.py
  ``` 
- Train the classification model:
  ```bash
  python MobileNetSSD/Training.py
  ```

### **3. Dataset Generation**

- Generate a dataset:
  ```bash
  python Image_Generation/creating_dataset.py
  ```

---

## **Troubleshooting**

- **Object Detection Issues:** Ensure the camera is connected, and the detection threshold is appropriate.
- **Robot Movement Inaccuracy:** Recalibrate the robot using `calibrate.py`.
- **Dataset Quality:** Ensure images are properly labeled and augmented.

---

## **Future Improvements**

- Advanced defect classification models for higher accuracy.
- Real-time feedback for robot operations.
- Improved object detection pipelines for diverse object types.

---

## **Contributors**

- [AniketMunde](https://github.com/AniketMunde)
- [Nirzar-shah-11](https://github.com/Nirzar-shah-11)
- [MasterMind232000](https://github.com/MasterMind232000)

---

## **License**

This project is licensed under the MIT License.

