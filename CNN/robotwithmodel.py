import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pyniryo import NiryoRobot
import time

# Load the bottle defect classification model
classifier_model = load_model('bottle_defect_classifier.h5')

# Load the pre-trained MobileNet-SSD model for bottle detection
net = cv2.dnn.readNetFromCaffe('CNN/deploy.prototxt', 'CNN/mobilenet_iter_73000.caffemodel')

# Define classes for MobileNet-SSD (21 classes)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Initialize the Niryo robot
robot = NiryoRobot("10.10.10.10")
robot.calibrate_auto()

# Define the pick and place positions
pick_position = [0.2, -0.1, 0.25]  # Adjust based on actual bottle position
place_position = [0.5, -0.1, 0.25]  # Fixed place position

# Function to preprocess images for defect classification
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img
# Function to uncompress an image from the robot's compressed image format
def uncompress_image(img_compressed):
    # Convert the compressed image to a NumPy array
    img_np = np.frombuffer(img_compressed, np.uint8)
    # Decode the image as an OpenCV image
    img_decoded = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    return img_decoded

# Start capturing images from the Niryo robot camera
while True:
    # Capture the compressed image from the robot camera
    img_compressed = robot.get_img_compressed()
    img_raw = uncompress_image(img_compressed)

    # Prepare the frame for bottle detection
    (h, w) = img_raw.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img_raw, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Process each detection and classify only bottles
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Adjust confidence threshold as needed
            idx = int(detections[0, 0, i, 1])

            # Check if detected object is a bottle
            if CLASSES[idx] == "bottle":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Crop the detected bottle region
                bottle_region = img_raw[startY:endY, startX:endX]

                # Classify the bottle region as defective or non-defective
                processed_img = preprocess_image(bottle_region)
                prediction = classifier_model.predict(processed_img)
                label = "Defective" if prediction[0][0] > 0.5 else "Non-defective"
                color = (0, 0, 255) if label == "Defective" else (0, 255, 0)

                # Draw bounding box and label on the frame
                cv2.rectangle(img_raw, (startX, startY), (endX, endY), color, 2)
                cv2.putText(img_raw, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # If the bottle is defective, pick it up and place it at the fixed position
                if label == "Defective":
                    robot.move_to_neutral()
                    time.sleep(2)
                    robot.move_pose(pick_position[0], pick_position[1], pick_position[2], 0, 1.57, 0)
                    robot.grip(True)  # Assuming robot has a gripper function
                    time.sleep(1)

                    # Move to the place position
                    robot.move_pose(place_position[0], place_position[1], place_position[2], 0, 1.57, 0)
                    robot.grip(False)  # Release the defective bottle
                    time.sleep(1)

    # Show the frame with detected and classified bottles
    cv2.imshow("Bottle Detection and Classification", img_raw)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
robot.close()
