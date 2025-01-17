from pyniryo import *
import cv2
from ultralytics import YOLO
import numpy as np
import time
import math

# Load the trained YOLO model
model = YOLO("yolov8_custom1.pt")  # YOLOv8 model

logitech_camera = cv2.VideoCapture(0)  # External camera for vertical inspection

# Connect to the Niryo robot
robot = NiryoRobot("10.10.10.10")
work_space = "QualityInspection1"
robot.calibrate_auto()
robot.update_tool()

# Constants
WIDTH_THRESHOLD = 30
HEIGHT_THRESHOLD = 100

# Initialize global variable
height_offset_var = 0.0

# Define robot positions (retrieved from the previous code)
home_position = [-0.0722594338208351, 0.22974860652649942, -0.5355637851217977, 0.07525771219618926, -1.3085782656562448, -0.10421803998643053]  # Home position
observation_position = [0.03123160950542081, 0.3085257876445155, -0.4461819834686641, 0.05071401959001909, -1.610772480869716, 0.020034403832306147]  # Vision pick observation position
place_up_left = (1.5653341341063887, 0.3751834024366829, -0.5916167793788476, -0.04899473162254786, -1.4128889592324687, 0.030772269347505876)  # Drop defective bottles
place_up_right = (-1.5089586235265033, 0.4372961798566571, -0.5855569962159233, 0.03997615407481936, -1.3806753626868695, -0.006043269561749831)  # Drop good bottles

robot.move_joints(home_position)  # Move to home position

# Grabbing logic
def grabbing_bottle(drop_position,height_offset_var):
    """Move to observation, grab the bottle, and drop at specified position."""
    robot.release_with_tool()
    robot.move_joints(observation_position)
    max_catch_count = 2
    catch_count = 0
    while catch_count < max_catch_count:
        
        robot.release_with_tool()
        # Moving to observation pose
        robot.move_joints(observation_position)
        
        #trying to get an object via Vision pick
        obj_found, shape_ret, color_ret = robot.vision_pick(work_space,
                                                            height_offset=height_offset_var,
                                                            shape=ObjectShape.ANY,
                                                            color=ObjectColor.ANY,)
        # robot.move_joints(step_back_position)
        if not obj_found:
            robot.wait(0.05)
            continue
        else:
            #travel to drop pos 
            pos = drop_position
            robot.move_joints(pos)
            time.sleep(1)
            robot.release_with_tool()

        catch_count += 1
        print("Object Count:"+ str(catch_count))



def inspect_bottles(source, is_robot, duration=5):
    good_count, bad_count, unknown_count = 0, 0, 0
    start_time = time.time()

    while time.time() - start_time < duration:
        if is_robot:
            img_compressed = source.get_img_compressed()
            img_raw = uncompress_image(img_compressed)
            results = model(img_raw)
        else:
            ret, frame = source.read()
            if not ret:
                print("External camera not accessible.")
                break
            results = model(frame)

        for result in results:
            for box in result.boxes:

                if is_robot:
                    # Calculate bottle dimensions
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    width = x2 - x1
                    height = y2 - y1

                    # Check if the bottle size is acceptable
                    if width > WIDTH_THRESHOLD or height > HEIGHT_THRESHOLD:
                        print("Bottle size not matchable. Skipping.")
                        continue

                # Count bottles based on label
                label = int(box.cls)
                if label == 0:
                    good_count += 1
                elif label == 1:
                    bad_count += 1
                else:
                    unknown_count += 1

    print(f"Good: {good_count}, Bad: {bad_count}, Unknown: {unknown_count}")

    return good_count,bad_count,unknown_count



# ==============================================================
# Horizontal inspection (Robot camera)
# ==============================================================

print("Starting horizontal inspection...")
good_count,bad_count,unknown_count = inspect_bottles(robot, is_robot=True)

if unknown_count > good_count or unknown_count > bad_count:
    height_offset_var = 0.02
    if good_count > bad_count:
        print("Good bottle detected consistently over 10 seconds.")
        grabbing_bottle(place_up_right,height_offset_var)
        robot.close_connection()
        logitech_camera.release()
    elif bad_count > good_count:
        print("Bad bottle detected consistently over 10 seconds.")
        grabbing_bottle(place_up_left,height_offset_var)
        robot.close_connection()
        logitech_camera.release()
else:
    print("Starting vertical inspection...")
# ==============================================================
# Vertical inspection (Logitech camera)
# ==============================================================

good_count,bad_count,unknown_count= inspect_bottles(logitech_camera, is_robot=False)
if good_count > bad_count:
    print("Good bottle detected consistently over 10 seconds.")
    grabbing_bottle(place_up_right,height_offset_var)
    robot.close_connection()
    logitech_camera.release()
elif bad_count > good_count:
    print("Bad bottle detected consistently over 10 seconds.")
    grabbing_bottle(place_up_left,height_offset_var)
    robot.close_connection()
    logitech_camera.release()
