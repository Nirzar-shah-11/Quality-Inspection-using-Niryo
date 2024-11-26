import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the bottle defect classification model
classifier_model = load_model('bottle_defect_classifier.h5')

# Load the pre-trained MobileNet-SSD model for bottle detection-it
net = cv2.dnn.readNetFromCaffe('CNN/deploy.prototxt', 'CNN/mobilenet_iter_73000.caffemodel')

# Define classes for MobileNet-SSD (21 classes)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Function to preprocess images for defect classification
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for bottle detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
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
                bottle_region = frame[startY:endY, startX:endX]

                # Classify the bottle region as defective or non-defective
                processed_img = preprocess_image(bottle_region)
                prediction = classifier_model.predict(processed_img)
                label = "Defective" if prediction[0][0] > 0.5 else "Non-defective"
                color = (0, 0, 255) if label == "Defective" else (0, 255, 0)

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame with detected and classified bottles
    cv2.imshow("Bottle Detection and Classification", frame)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
