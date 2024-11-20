from pyniryo import *
from ultralytics import YOLO
import cv2

# Connect to the robot
robot = NiryoRobot("10.10.10.10")
robot.calibrate_auto()

# Load YOLO model
yolo = YOLO('yolov8s.pt')

# Move the robot to the observation pose
robot.move_pose(0.2, -0.1, 0.25, 0.0, 1.57, 0.0)

# Function to get colors for each class
def get_colours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

while True:
    # Capture the compressed image from the robot
    img_compressed = robot.get_img_compressed()
    img_raw = uncompress_image(img_compressed)
    
    # Flip the image upside down
    img_flipped = cv2.flip(img_raw, 0)
    
    # Run YOLO detection on the flipped image
    results = yolo(img_flipped)

    # Draw the detected bounding boxes and labels
    for result in results:
        for box in result.boxes:
            if box.conf[0] > 0.4:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
                cls = int(box.cls[0])  # Class index
                class_name = result.names[cls]  # Class name
                if class_name.lower() == "bottle":
                    colour = get_colours(cls)  # Color for the class

                    # Draw rectangle and label on the image
                    cv2.rectangle(img_flipped, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(img_flipped, f'{class_name} {box.conf[0]:.2f}', 
                                (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

                    # Check if the detected object is a "bottle"
                    if class_name.lower() == "bottle":
                        # Crop the detected bottle area from the flipped image
                        bottle_image = img_flipped[y1:y2, x1:x2]

                        # Save the cropped image
                        cv2.imwrite("detected_bottle.jpg", bottle_image)
                        print("Bottle detected and saved as 'detected_bottle.jpg'")

    # Display the processed image
    key = show_img("Robot View", img_flipped, wait_ms=30)
    if key in [27, ord("q")]:  # Exit on pressing 'Esc' or 'Q'
        break
