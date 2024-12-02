from pyniryo import *
import cv2
from ultralytics import YOLO
import numpy as np
import time
import math

# Load the trained YOLO model
model = YOLO("yolov8_custom.pt")  # Replace with the path to your trained YOLOv8 model

# Connect to the Niryo robot
robot = NiryoRobot("10.10.10.10")
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
detection_position = (0.012968484212552145, 0.16157604594360092, -1.34, 0.11974315504487265, 0.5582763532005806, 0.004694595953449898)
grabbing_pos = (convert_data(top_position))
robot.move_joints(detection_position)
# Define Dropping position

# Place the bottle on right side
place_up_left = (1.5653341341063887, 0.3751834024366829, -0.5916167793788476, -0.04899473162254786, -1.4128889592324687, 0.030772269347505876)

# Place the bottle on left side
place_up_right = (-1.5089586235265033, 0.4372961798566571, -0.5855569962159233, 0.03997615407481936, -1.3806753626868695, -0.006043269561749831)


# Grabbing logic
def grabbing_left():
    # Initialise variables
    max_catch_count = 1
    catch_count = 0

    while catch_count < max_catch_count:
        robot.release_with_tool()
        # Moving to observation pose
        robot.move_pose(grabbing_pos)

        #trying to get an object via Vision pick
        obj_found, shape_ret, color_ret = robot.vision_pick(work_space,
                                                            height_offset=0.19,
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
        robot.move_pose(grabbing_pos)


        #trying to get an object via Vision pick
        obj_found, shape_ret, color_ret = robot.vision_pick(work_space,
                                                            height_offset=0.19,
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

#  Approach 2
while True:
    # Initialize counters for class labels
    good_count = 0
    bad_count = 0
    unknown_count = 0
    start_time = time.time()
    
    while time.time() - start_time < 10:  # Run YOLO for 3 seconds
        # Capture the compressed image from the robot
        img_compressed = robot.get_img_compressed()
        img_raw = uncompress_image(img_compressed)
        
        # Flip the image upside down
        img_flipped = cv2.flip(img_raw, 0)  # 0 for vertical flip

        # Perform YOLO inference on the flipped image
        results = model(img_raw)

        for result in results:
            for box in result.boxes:
                # Get the class label (0 for 'good', 1 for 'bad', assuming two classes)
                label = int(box.cls)
                if label == 0:
                    good_count += 1
                elif label == 1:
                    bad_count += 1
                else:
                    unknown_count += 1
    
    # Determine the final decision based on the counts
    if good_count > bad_count:
        print("Good can detected consistently over 10 seconds")
        grabbing_right()
        # robot.release_with_tool()
        # # Moving to observation pose]
        # robot.move_pose(grabbing_pos)

        pos = place_up_right
        robot.move_joints(pos)
        robot.release_with_tool()
        break

    elif bad_count > good_count:
        print("Bad can detected consistently over 10 seconds")
        grabbing_left()
        # pos = place_up_left
        # robot.move_joints(pos)
        # robot.release_with_tool()
    else:
        print("Unclear classification, skipping this round.")

    # Display the last captured image with YOLO detections
    for result in results:
        result_img = result.plot()  # Get the image with bounding boxes and labels
    key = show_img("Robot View with YOLO Detection", result_img, wait_ms=30)
    
    if key in [27, ord("q")]:  # Exit on pressing 'Esc' or 'Q'
        break

# Disconnect from the robot
robot.close_connection()


