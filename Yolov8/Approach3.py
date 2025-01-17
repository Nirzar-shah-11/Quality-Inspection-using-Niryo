from pyniryo import *
import cv2
from ultralytics import YOLO
import numpy as np
import time
import math

# Load the trained YOLO model
model = YOLO("yolov8_custom1.pt")  # YOLOv8 model

logitech_camera = cv2.VideoCapture(0) # Camera 2

# Connect to the Niryo robot
robot = NiryoRobot("10.10.10.10")
work_space = "QualityInspection1"
robot.calibrate_auto()
robot.update_tool()

# Define robot position 
observation_position = (0.008402702889334979, 0.08582875640704701, -0.3886140434208831, 0.0031606151655640957, -1.411354978444583, 0.04764605801424793) #vision pick observation position
detection_position = (0.10580603778463438, 0.2933763297372047, -1.34, 0.02770430777173427, 0.10575202077431634, 0.1396849052873863) # bottle observation position

# Define Dropping position
place_up_left = [1.519676320874217, 0.3009510586908601, -0.8112839190348539, -0.0443927892588909, -1.0830830898370558, -0.01831511586483492] # Place the bottle on right side
place_up_right = [-1.5074366964187642, 0.5251630357190596, -0.9612635523172306, 0.02463634619596311, -1.1536462060797947, -0.009111231137520992] # Place the bottle on left side


# Grabbing logic
def grabbing_left():
    # Initialise variables
    max_catch_count = 1
    catch_count = 0

    while catch_count < max_catch_count:
        robot.release_with_tool()
        # Moving to observation pose
        robot.move_joints(observation_position)
        
        #trying to get an object via Vision pick
        obj_found, shape_ret, color_ret = robot.vision_pick(work_space,
                                                            height_offset=0.15,
                                                            shape=ObjectShape.ANY,
                                                            color=ObjectColor.ANY,)
        if not obj_found:
            robot.led_ring.solid([255, 255, 0]) # Change the status of LED
            robot.wait(0.05)
            continue
        else:
            #travel to drop pos 
            robot.led_ring.solid([0, 255, 0]) # Change the status of LED
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
        robot.move_joints(observation_position)


        #trying to get an object via Vision pick
        obj_found, shape_ret, color_ret = robot.vision_pick(work_space,
                                                            height_offset=0.15,
                                                            shape=ObjectShape.ANY,
                                                            color=ObjectColor.ANY)
        if not obj_found:
            robot.led_ring.solid([255, 255, 0]) # Change the status of LED
            robot.wait(0.05)
            continue
        else:
            #travel to drop pos 
            robot.led_ring.solid([0, 255, 0]) # Change the status of LED
            pos = place_up_right
            robot.move_joints(pos)
            time.sleep(1)
            robot.release_with_tool()

        catch_count += 1
        print("Object Count:"+ str(catch_count))

robot.move_joints(detection_position)

while True:
    # Initialize counters for class labels
    good_count = 0
    bad_count = 0
    unknown_count = 0
    start_time = time.time()
    
    while time.time() - start_time < 10:  # Run YOLO for 10 seconds
        # ==============================================================================================================================
        # Camera 1
        # Capture the compressed image from the robot
        img_compressed = robot.get_img_compressed()
        img_raw = uncompress_image(img_compressed)
        # Flip the image upside down
        img_flipped = cv2.flip(img_raw, 0)  # 0 for vertical flip

        # Perform YOLO inference on the flipped image
        result_camera1 = model(img_raw)

        for result1 in result_camera1:
            for box in result1.boxes:
                # Get the class label (0 for 'good', 1 for 'bad', assuming two classes)
                label = int(box.cls)
                if label == 0:
                    good_count += 1
                elif label == 1:
                    bad_count += 1
                else:
                    unknown_count += 1
        
        # ==============================================================================================================================
       
        # ==============================================================================================================================
        # Camera 2
        # Capture image from external camera. 
        ret, frame = logitech_camera.read()
        if not ret:
            break
        # Perform inference on the webcam frame
        result_camera2 = model(frame)

        for result2 in result_camera2:
            for box in result2.boxes:
                # Get the class label (0 for 'good', 1 for 'bad', assuming two classes)
                label = int(box.cls)
                if label == 0:
                    good_count += 1
                elif label == 1:
                    bad_count += 1
                else:
                    unknown_count += 1

        # ==============================================================================================================================

    # Determine the final decision based on the counts
    if good_count > bad_count:
        print("Good can detected consistently over 10 seconds")
        grabbing_right()
        break
    elif bad_count > good_count:
        print("Bad can detected consistently over 10 seconds")
        grabbing_left()
        break
    else:
        robot.led_ring.solid([255, 0, 0]) # Change the status of LED
        print("Unclear classification, skipping this round.")
        

    # Display the last captured image with YOLO detections
    for result in result_camera1:
        result_img1 = result.plot()  # Get the image with bounding boxes and labels
    key = show_img("Robot View with YOLO Detection", result_img1, wait_ms=30)
    
    for result in result_camera2:
        result_img2 = result.plot()  # Get the image with bounding boxes and labels
    key = show_img("Robot View with YOLO Detection", result_img2, wait_ms=30)

    if key in [27, ord("q")]:  # Exit on pressing 'Esc' or 'Q'
        break

# Disconnect from the robot
robot.close_connection()


