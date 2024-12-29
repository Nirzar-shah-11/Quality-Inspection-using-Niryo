from pyniryo import *
import cv2
import numpy as np
import time
import math

# Load the trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe(
    'mobilenet+CNN/CNN/deploy.prototxt',
    'mobilenet+CNN/CNN/mobilenet_iter_73000.caffemodel'
)

# Connect to the Niryo robot
robot = NiryoRobot("10.10.10.10")
work_space = "QualityInspection"
robot.calibrate_auto()
robot.update_tool()

def convert_data(data):
    pos_cm = np.array([data[0], data[1], data[2]])
    pos_m = np.array(pos_cm / 100)
    pos_m = np.around(pos_m, 2)
    end_d = np.array([data[3], data[4], data[5]])
    end_r = np.array(end_d) * (math.pi / 180)
    end_r = np.around(end_r, 2)

    # Sort data
    j1 = pos_m[0]
    j2 = pos_m[1]
    j3 = pos_m[2]
    j4 = end_r[0]
    j5 = end_r[1]
    j6 = end_r[2]

    convert = (j1, j2, j3, j4, j5, j6)
    return convert

# Define robot positions
home = [25, 0, 30, 0, 90, 0]  # Observation position
top_position = [25, 0, 40, 0, 90, 0]  # Top observation position
detection_position = (0.012968484212552145, 0.16157604594360092, -1.34, 0.11974315504487265, 0.5582763532005806, 0.004694595953449898)
grabbing_pos = convert_data(top_position)
robot.move_joints(detection_position)

# Define dropping positions
place_up_left = (1.5653341341063887, 0.3751834024366829, -0.5916167793788476, -0.04899473162254786, -1.4128889592324687, 0.030772269347505876)
place_up_right = (-1.5089586235265033, 0.4372961798566571, -0.5855569962159233, 0.03997615407481936, -1.3806753626868695, -0.006043269561749831)

# Preprocess the image for MobileNet classification
def preprocess_image(img):
    img = cv2.resize(img, (300, 300))
    blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
    return blob

# Grabbing logic
def grabbing_left():
    max_catch_count = 1
    catch_count = 0

    while catch_count < max_catch_count:
        robot.release_with_tool()
        robot.move_pose(grabbing_pos)

        obj_found, shape_ret, color_ret = robot.vision_pick(work_space, height_offset=0.19, shape=ObjectShape.ANY, color=ObjectColor.ANY)
        if not obj_found:
            robot.wait(0.1)
            continue
        else:
            robot.move_joints(place_up_left)
            time.sleep(1)
            robot.release_with_tool()

        catch_count += 1
        print("Object Count:" + str(catch_count))

def grabbing_right():
    max_catch_count = 1
    catch_count = 0

    while catch_count < max_catch_count:
        robot.release_with_tool()
        robot.move_pose(grabbing_pos)

        obj_found, shape_ret, color_ret = robot.vision_pick(work_space, height_offset=0.19, shape=ObjectShape.ANY, color=ObjectColor.ANY)
        if not obj_found:
            robot.wait(0.1)
            continue
        else:
            robot.move_joints(place_up_right)
            time.sleep(1)
            robot.release_with_tool()

        catch_count += 1
        print("Object Count:" + str(catch_count))

# Main detection and classification loop
while True:
    good_count = 0
    bad_count = 0
    start_time = time.time()

    while time.time() - start_time < 10:  # Run detection for 10 seconds
        img_compressed = robot.get_img_compressed()
        img_raw = uncompress_image(img_compressed)

        blob = preprocess_image(img_raw)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:  # Confidence threshold
                idx = int(detections[0, 0, i, 1])
                label = "good" if idx == 0 else "bad"  # Assume 0: good, 1: bad

                if label == "good":
                    good_count += 1
                elif label == "bad":
                    bad_count += 1

    if good_count > bad_count:
        print("Good bottle detected consistently over 10 seconds")
        grabbing_right()
        robot.move_joints(place_up_right)
        robot.release_with_tool()
        break

    elif bad_count > good_count:
        print("Bad bottle detected consistently over 10 seconds")
        grabbing_left()
    else:
        print("Unclear classification, skipping this round.")

    # Display detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            box = detections[0, 0, i, 3:7] * np.array([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            label = "good" if idx == 0 else "bad"
            color = (0, 255, 0) if label == "good" else (0, 0, 255)
            cv2.rectangle(img_raw, (startX, startY), (endX, endY), color, 2)
            cv2.putText(img_raw, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    key = cv2.waitKey(30) & 0xFF
    if key in [27, ord("q")]:
        break

robot.close_connection()
