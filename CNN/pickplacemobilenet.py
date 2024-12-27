from pyniryo import *
import cv2
import numpy as np
import math
import time
from tensorflow.keras.models import load_model

# Robot connection details
work_space = "QualityInspection"
robot = NiryoRobot("10.10.10.10")

# Load MobileNet SSD model for detection
net = cv2.dnn.readNetFromCaffe('mobilenet+CNN/CNN/deploy.prototxt', 'mobilenet+CNN/CNN/mobilenet_iter_73000.caffemodel')

# Load the defect classifier
classifier_model = load_model('mobilenet+CNN/bottle_defect_classifier.h5')

# Classes for MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Function to convert positional data
def convert_data(data):
    pos_cm = np.array([data[0], data[1], data[2]])
    pos_m = pos_cm / 100
    pos_m = np.around(pos_m, 2)
    end_d = np.array([data[3], data[4], data[5]])
    end_r = end_d * (math.pi / 180)
    end_r = np.around(end_r, 2)

    return (*pos_m, *end_r)

# Function to preprocess images for defect classification
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Initialize robot and calibrate
robot.calibrate_auto()
robot.update_tool()

# Predefined positions
home = ([25, 0, 30, 0, 90, 0])  # Home position
obs = ([25, 0, 40, 0, 90, 0])  # Observation position
place_up = ([0, 17, 25, 0, 90, 90])
place_down = ([0, 17, 9, 0, 90, 90])

# Start the pick-and-place loop
catch_count = 0
max_catch_count = 1

while catch_count < max_catch_count:
    # Move to observation position
    robot.move_pose(*convert_data(obs))

    # Capture image from robot camera
    img_compressed = robot.get_img_compressed()
    img_raw = cv2.imdecode(np.frombuffer(img_compressed, np.uint8), cv2.IMREAD_COLOR)

    # Detect bottles using MobileNet SSD
    (h, w) = img_raw.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img_raw, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    bottle_detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Adjust confidence threshold
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

                if label == "Defective":
                    # Pick the defective bottle
                    pick_position = [
                        (startX + endX) / 2 / 100,  # X position in meters
                        (startY + endY) / 2 / 100,  # Y position in meters
                        0.10,                      # Z position (slightly above the object)
                        0, 90, 0
                    ]

                    robot.move_pose(*convert_data(pick_position))  # Move to the pick position
                    time.sleep(1)
                    robot.grasp_with_tool()
                    time.sleep(1)

                    # Place the defective bottle
                    robot.move_pose(*convert_data(place_up))
                    time.sleep(1)
                    robot.move_pose(*convert_data(place_down))
                    robot.release_with_tool()
                    time.sleep(1)

                # Break the loop after processing one bottle
                break

    if not bottle_detected:
        print("No bottle detected. Retrying...")
        time.sleep(1)
        continue

    catch_count += 1
    print("Processed object count:", catch_count)

# Move to home position after completing tasks
robot.move_pose(*convert_data(home))
robot.close_connection()
