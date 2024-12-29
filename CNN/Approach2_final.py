from pyniryo import *
import cv2
import numpy as np
import time
import math
from keras.models import load_model

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

# Initialize the external camera for vertical inspection
logitech_camera = cv2.VideoCapture(0)

# Connect to the Niryo robot
robot = NiryoRobot("10.10.10.10")
work_space = "QualityInspection1"
robot.calibrate_auto()
robot.update_tool()

# Define robot positions
home_position = (-0.0098, 0.3362, -0.4084, 0.0422, 1.2220, -0.0298)  # Home position
observation_position = (0.008402702889334979, -0.0065829368275487354, -0.32195642862871576, -0.004509288773864029, -1.3131802080199013, -0.10421803998643053)  # Vision pick observation position
place_up_left = (1.5653341341063887, 0.3751834024366829, -0.5916167793788476, -0.04899473162254786, -1.4128889592324687, 0.030772269347505876)  # Drop defective bottles
place_up_right = (-1.5089586235265033, 0.4372961798566571, -0.5855569962159233, 0.03997615407481936, -1.3806753626868695, -0.006043269561749831)  # Drop good bottles

def grabbing_bottle(drop_position):
    """Move to observation, grab the bottle, and drop at specified position."""
    robot.release_with_tool()
    robot.move_joints(observation_position)

    # Try to pick the object
    obj_found, shape_ret, color_ret = robot.vision_pick(
        work_space, height_offset=0.01, shape=ObjectShape.ANY, color=ObjectColor.ANY
    )
    if not obj_found:
        print("No object found. Skipping.")
        return False

    # Travel to drop position
    robot.move_joints(drop_position)
    time.sleep(1)
    robot.release_with_tool()
    return True

# Start operations
robot.move_joints(home_position)  # Move to home position

while True:
    good_count, bad_count = 0, 0
    start_time = time.time()

    while time.time() - start_time < 10:  # Run for 10 seconds
        # ==============================================================
        # Horizontal inspection (Robot camera)
        img_compressed = robot.get_img_compressed()
        img_raw = uncompress_image(img_compressed)

        # Prepare input for MobileNet SSD
        blob = cv2.dnn.blobFromImage(img_raw, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                class_id = int(detections[0, 0, i, 1])
                label = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"
                if label == "bottle":
                    # Crop the detected bottle and pass it to the defect classifier
                    x1, y1, x2, y2 = (detections[0, 0, i, 3:] * np.array([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]])).astype("int")
                    cropped_bottle = img_raw[y1:y2, x1:x2]
                    cropped_bottle = cv2.resize(cropped_bottle, (224, 224))  # Resize for defect classifier
                    cropped_bottle = cropped_bottle.astype("float32") / 255.0
                    cropped_bottle = np.expand_dims(cropped_bottle, axis=0)

                    # Predict defect
                    defect_prediction = classifier_model.predict(cropped_bottle)
                    if defect_prediction[0][0] > 0.5:  # Assuming binary classification: 0 = good, 1 = defective
                        bad_count += 1
                    else:
                        good_count += 1

        # ==============================================================
        # Vertical inspection (External camera)
        ret, frame = logitech_camera.read()
        if not ret:
            print("External camera not accessible.")
            break

        # Repeat similar MobileNet SSD logic for external camera
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                label = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"
                if label == "bottle":
                    x1, y1, x2, y2 = (detections[0, 0, i, 3:] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype("int")
                    cropped_bottle = frame[y1:y2, x1:x2]
                    cropped_bottle = cv2.resize(cropped_bottle, (224, 224))
                    cropped_bottle = cropped_bottle.astype("float32") / 255.0
                    cropped_bottle = np.expand_dims(cropped_bottle, axis=0)

                    defect_prediction = classifier_model.predict(cropped_bottle)
                    if defect_prediction[0][0] > 0.5:
                        bad_count += 1
                    else:
                        good_count += 1

    # Decision based on counts
    if good_count > bad_count:
        print("Good bottle detected consistently over 10 seconds.")
        grabbing_bottle(place_up_right)
        break
    elif bad_count > good_count:
        print("Bad bottle detected consistently over 10 seconds.")
        grabbing_bottle(place_up_left)
    else:
        print("Unclear classification. Skipping this round.")

# Disconnect robot and release camera
robot.close_connection()
logitech_camera.release()
