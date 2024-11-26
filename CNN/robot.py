import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pyniryo import *

# Load the trained model
model = load_model('bottle_defect_classifier.h5')

# Connect to the robot
robot = NiryoRobot("10.10.10.10")
robot.calibrate_auto()

# Define positions for picking and placing defective bottles
pick_positions = [
    (0.1, -0.15, 0.05, 0.0, 1.57, 0.0),  # position 1
    (0.2, -0.1, 0.05, 0.0, 1.57, 0.0),   # position 2
    (0.1, -0.05, 0.05, 0.0, 1.57, 0.0),  # position 3
    (0.2, -0.2, 0.05, 0.0, 1.57, 0.0)    # position 4
]
place_position = (0.3, 0.2, 0.05, 0.0, 1.57, 0.0)

# Function to preprocess images for model input
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))  # Resize to model's expected input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Move to observation pose and capture image
robot.move_pose(0.2, -0.1, 0.25, 0.0, 1.57, 0.0)

# Capture an image
img_compressed = robot.get_img_compressed()
img_raw = uncompress_image(img_compressed)
img_flipped = cv2.flip(img_raw, 0)

# Placeholder: Segment the image and classify each bottle (assuming 4 locations)
for idx, pos in enumerate(pick_positions):
    # Extract bottle image region based on pre-defined regions or detected locations
    bottle_img = img_flipped[50:150, 50*(idx+1):150*(idx+1)]  # Example segmentation

    # Preprocess and classify
    processed_img = preprocess_image(bottle_img)
    prediction = model.predict(processed_img)

    if prediction[0][0] > 0.5:  # If predicted as defective
        # Move to the bottle's position
        robot.move_pose(*pos)

        # Pick the bottle
        robot.close_gripper()

        # Move to the place position
        robot.move_pose(*place_position)

        # Release the bottle
        robot.open_gripper()

# Return to observation pose or home position
robot.move_pose(0.2, -0.1, 0.25, 0.0, 1.57, 0.0)

# Disconnect from the robot
robot.end()
