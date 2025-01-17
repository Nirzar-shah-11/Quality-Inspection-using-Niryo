from pyniryo import *
import cv2
import numpy as np
import math
import time
from tensorflow.keras.models import load_model

# Robot connection details
robot = NiryoRobot("10.10.10.10")  # Replace with your robot's IP
workspace = "QualityInspection1"
robot.calibrate_auto()
robot.update_tool()

# Load MobileNet SSD model for detection
net = cv2.dnn.readNetFromCaffe(
    'mobilenet+CNN/CNN/deploy.prototxt',
    'mobilenet+CNN/CNN/mobilenet_iter_73000.caffemodel'
)

# Load the defect classifier model
classifier_model = load_model('mobilenet+CNN/bottle_defect_classifier.h5')

# Classes for MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Function to convert position data
def convert_data(data):
    pos_cm = np.array([data[0], data[1], data[2]])
    pos_m = pos_cm / 100
    pos_m = np.around(pos_m, 2)
    end_d = np.array([data[3], data[4], data[5]])
    end_r = end_d * (math.pi / 180)
    end_r = np.around(end_r, 2)
    return (*pos_m, *end_r)

# Preprocess images for defect classification
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Predefined positions
home = [25, 0, 30, 0, 90, 0]  # Home position
top_position = [25, 0, 40, 0, 90, 0]  # Top observation position
place_up_left = [1.57, 0.34, -0.82, 0.0, -1.09, 0.0]  # Non-defective drop
place_up_right = [-1.57, 0.34, -0.82, 0.0, -1.09, 0.0]  # Defective drop
pick_offset_z = -0.02  # Z-axis offset for picking

# Ensure the robot starts at the home position
robot.move_pose(*convert_data(home))

while True:
    # Move to home position before detection
    robot.move_pose(*convert_data(home))
    time.sleep(1)

    # Capture image from the robot
    img_compressed = robot.get_img_compressed()
    img_raw = uncompress_image(img_compressed)

    # Detect objects using MobileNet SSD
    (h, w) = img_raw.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img_raw, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    bottle_detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "bottle":
                bottle_detected = True

                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                bottle_region = img_raw[startY:endY, startX:endX]

                # Classify bottle
                processed_img = preprocess_image(bottle_region)
                prediction = classifier_model.predict(processed_img)
                label = "Defective" if prediction[0][0] > 0.5 else "Non-defective"

                # Move to top observation position
                robot.move_pose(*convert_data(top_position))
                time.sleep(1)

                # Adjust Z-axis for pick position
                pick_position = [25, 0, 38, 0, 90, 0]  # Move closer to the bottle
                pick_position[2] += pick_offset_z * 100  # Apply Z-offset
                robot.move_pose(*convert_data(pick_position))
                time.sleep(1)

                # Grasp the bottle
                robot.grasp_with_tool()

                # Act based on classification
                if label == "Defective":
                    print("Defective bottle detected")
                    robot.move_joints(*convert_data(place_up_right))
                    time.sleep(1)
                    robot.release_with_tool()
                else:
                    print("Non-defective bottle detected")
                    robot.move_joints(*convert_data(place_up_left))
                    time.sleep(1)
                    robot.release_with_tool()

                # Return to home position
                robot.move_pose(*convert_data(home))
                break

    if not bottle_detected:
        print("No bottle detected. Retrying...")
        time.sleep(1)
        continue
