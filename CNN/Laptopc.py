import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('bottle_defect_classifier.h5')

# Function to preprocess images for model input
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))  # Resize to match the input shape of the model
    img = img / 255.0  # Normalize pixel values between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Initialize the laptop camera
cap = cv2.VideoCapture(0)  # 0 is the default camera

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if there’s an issue with the camera

    # Display instructions on the screen
    cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Assume there are four bottles in the frame; adjust coordinates as needed
    # Here we divide the frame into four regions (example segmentation)
    height, width, _ = frame.shape
    bottle_regions = [
        frame[0:height, 0:width//4],         # Leftmost quarter
        frame[0:height, width//4:width//2],  # Second quarter
        frame[0:height, width//2:3*width//4],# Third quarter
        frame[0:height, 3*width//4:width]    # Rightmost quarter
    ]

    # Loop through each bottle region, classify, and draw the result
    for i, bottle_img in enumerate(bottle_regions):
        # Preprocess each bottle image and make a prediction
        processed_img = preprocess_image(bottle_img)
        prediction = model.predict(processed_img)

        # Determine label based on prediction
        label = "Defective" if prediction[0][0] > 0.5 else "Non-defective"
        color = (0, 0, 255) if label == "Defective" else (0, 255, 0)

        # Draw the label on the frame
        x_start = i * width // 4
        cv2.putText(frame, label, (x_start + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (x_start, 0), (x_start + width // 4, height), color, 2)

    # Show the frame with classifications
    cv2.imshow("Bottle Classification", frame)


    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
