# ==============================================================================================================================
# Yolo with Niryo's camera 
# ==============================================================================================================================

from pyniryo import *
import cv2
from ultralytics import YOLO
import numpy as np
import time
import math

# Load the trained YOLO model
model = YOLO("yolov8_custom.pt")  # Replace with the path to your trained YOLOv8 model

# Connect to the Niryo robot
robot = NiryoRobot("*.*.*.*")
work_space = "QualityInspection"
robot.calibrate_auto()
robot.update_tool()

def convert_data(data):
    pos_cm = np.array([data[0],data[1],data[2]])
    pos_m = np.array(pos_cm/100)
    pos_m = np.around(pos_m,2)
    end_d = np.array([data[3],data[4],data[5]])
    end_r = np.array(end_d) * (math.pi / 180)
    end_r = np.around(end_r,2)

    #sort data
    j1 = pos_m[0]
    j2 = pos_m[1]
    j3 = pos_m[2]
    j4 = end_r[0]
    j5 = end_r[1]
    j6 = end_r[2]

    convert = (j1,j2,j3,j4,j5,j6)
    return (convert)

# Define robot position 
home = ([25,0,30,0,90,0]) #observation
top_position = ([25,0,40,0,90,0]) #observation
pos = (convert_data(top_position))

# Define Dropping position
# Place the good bottle on right side
place_up_left = (1.5698999154296058, 0.3403396492498681, -0.8234034853607025, 0.0016266343776782932, -1.090752993776484, 9.265358979293481e-05)

# Place the bad bottle on left side
place_up_right = (-1.5698357078360656, 0.3403396492498681, -0.8234034853607025, 0.0016266343776782932, -1.0892190129885981, 9.265358979293481e-05)


# Grabbing logic
def grabbing_left():
    # Initialise variables
    max_catch_count = 1
    catch_count = 0

    while catch_count < max_catch_count:
        robot.release_with_tool()
        # Moving to observation pose
        robot.move_pose(pos)

        #trying to get an object via Vision pick
        obj_found, shape_ret, color_ret = robot.vision_pick(work_space,
                                                            height_offset=0.05,
                                                            shape=ObjectShape.ANY,
                                                            color=ObjectColor.ANY)
        if not obj_found:
            robot.wait(0.1)
            continue
        else:
            #travel to drop pos 
            pos = place_up_left
            robot.move_joints(pos)
            time.sleep(1)
            robot.release_with_tool()

        catch_count += 1
        print("Object Count:"+ str(catch_count))

def grabbing_right():
    # Initialise variables
    max_catch_count = 1
    catch_count = 0

    while catch_count < max_catch_count:
        robot.release_with_tool()
        # Moving to observation pose]
        robot.move_pose(pos)


        #trying to get an object via Vision pick
        obj_found, shape_ret, color_ret = robot.vision_pick(work_space,
                                                            height_offset=0.05,
                                                            shape=ObjectShape.ANY,
                                                            color=ObjectColor.ANY)
        if not obj_found:
            robot.wait(0.1)
            continue
        else:
            #travel to drop pos 
            pos = place_up_right
            robot.move_joints(pos)
            time.sleep(1)
            robot.release_with_tool()

        catch_count += 1
        print("Object Count:"+ str(catch_count))

# Move the robot to the observation pose

while True:
    # Capture the compressed image from the robot
    img_compressed = robot.get_img_compressed()
    img_raw = uncompress_image(img_compressed)
    
    # Flip the image upside down
    img_flipped = cv2.flip(img_raw, 0)  # 0 for vertical flip

    # Perform YOLO inference on the flipped image
    results = model(img_raw)

    # Loop through results and plot the bounding boxes on the image
    for result in results:

        for box in result.boxes:
            # Get the class label (0 for 'good', 1 for 'bad', assuming two classes)
            label = int(box.cls)
            if label == 0:
                print("Good can detected")
                grabbing_right()
            elif label == 1:
                print("Bad can detected")
                grabbing_left()
            else:
                print("Unknown object detected, skipping.")
        # Plot detections on the image
        result_img = result.plot()  # Get the image with bounding boxes and labels

    # Display the image with YOLO detections
    key = show_img("Robot View with YOLO Detection", result_img, wait_ms=30)
    if key in [27, ord("q")]:  # Exit on pressing 'Esc' or 'Q'
        break

# Disconnect from the robot
robot.end() 
