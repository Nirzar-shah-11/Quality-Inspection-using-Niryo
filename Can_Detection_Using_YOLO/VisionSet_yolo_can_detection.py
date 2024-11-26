# ==============================================================================================================================
# Yolo with Niryo's camera 
# ==============================================================================================================================

from pyniryo import *
import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("yolov8_custom.pt")  # Replace with the path to your trained YOLOv8 model

# Connect to the Niryo robot
robot = NiryoRobot("10.10.10.10")
robot.calibrate_auto()
robot.update_tool()

# Move the robot to the observation pose
robot.move_joints(0.10124025646141721, 0.163090991734332, -1.3112160299761095, 0.1504227708025856, 0.5782181034430938, -0.30670350398733515)


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
        # Plot detections on the image
        result_img = result.plot()  # Get the image with bounding boxes and labels

    # Display the image with YOLO detections
    key = show_img("Robot View with YOLO Detection", result_img, wait_ms=30)
    if key in [27, ord("q")]:  # Exit on pressing 'Esc' or 'Q'
        break

# Disconnect from the robot
robot.end()
