from pyniryo import *
import cv2
import numpy as np
import math
import time

# Load the MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe("mobilenet+CNN\\CNN\\deploy.prototxt", "mobilenet+CNN\\CNN\\mobilenet_iter_73000.caffemodel")

# Define classes for MobileNet-SSD (21 classes)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Connect to the Niryo robot
robot = NiryoRobot("10.10.10.10")  # Replace with robot IP
robot.calibrate_auto()
robot.update_tool()

# Load the workspace bounds file
def load_workspace_bounds(file_path):
    workspace_bounds = {}
    with open(file_path, "r") as file:
        for line in file:
            corner, pose = line.strip().split(": ")
            # Parse pose manually since it contains `x = ..., y = ..., z = ...` format
            pose_dict = {}
            for item in pose.split(", "):
                key, value = item.split(" = ")
                pose_dict[key] = float(value)
            workspace_bounds[corner] = (
                pose_dict["x"],
                pose_dict["y"],
                pose_dict["z"],
                pose_dict["roll"],
                pose_dict["pitch"],
                pose_dict["yaw"]
            )
    return workspace_bounds

workspace_file = "QualityInspection_workspace_bounds.txt"
workspace_bounds = load_workspace_bounds(workspace_file)

# Define positions for drop based on workspace bounds
place_up_left = PoseObject(*workspace_bounds["top_left"])  # Left drop for defective
place_up_right = PoseObject(*workspace_bounds["top_right"])  # Right drop for non-defective

# Define a home position (optional, ensure it's within the workspace bounds)
home_pose = PoseObject(*workspace_bounds["bottom_left"])  # Example: bottom-left as a home position

# Preprocessing function for object detection
def preprocess_frame(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    return blob, h, w

# Start processing frames
while True:
    # Capture an image from the Niryo camera
    img_compressed = robot.get_img_compressed()
    img_raw = cv2.imdecode(np.frombuffer(img_compressed, np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the frame for MobileNet-SSD
    blob, h, w = preprocess_frame(img_raw)
    net.setInput(blob)
    detections = net.forward()

    # Process detections
    detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "bottle":
                detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw bounding box
                cv2.rectangle(img_raw, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(img_raw, "Bottle", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Compute center of the detected bottle
                pick_x = ((startX + endX) / 2) / 1000.0
                pick_y = ((startY + endY) / 2) / 1000.0
                pick_z = 0.15  # Fixed height for picking

                # Simulate classification logic (e.g., based on bounding box size, position)
                if endX - startX > 50:  # Example threshold for "defective" classification
                    print("Defective bottle detected")
                    robot.move_pose(PoseObject(pick_x, pick_y, pick_z, 0.0, 1.57, 0.0))
                    robot.grasp_with_tool()
                    robot.move_pose(place_up_left)  # Drop defective on left
                    robot.release_with_tool()
                else:
                    print("Non-defective bottle detected")
                    robot.move_pose(PoseObject(pick_x, pick_y, pick_z, 0.0, 1.57, 0.0))
                    robot.grasp_with_tool()
                    robot.move_pose(place_up_right)  # Drop non-defective on right
                    robot.release_with_tool()

    if not detected:
        print("No bottles detected")

    # Display the current frame
    cv2.imshow("Bottle Detection", img_raw)

    # Exit on 'q' or 'Esc'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

# Cleanup
cv2.destroyAllWindows()
robot.close_connection()
