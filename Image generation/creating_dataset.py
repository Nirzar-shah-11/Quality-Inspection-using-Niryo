import cv2
import time
import os
from datetime import datetime

# Set up a directory to save images
save_dir = " "
os.makedirs(save_dir, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Capture images every 10 seconds
last_capture_time = time.time()

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture frame.")
            break
        
        # Display the live video feed
        cv2.imshow("Live Feed", frame)
        
        # Capture an image every 10 seconds
        current_time = time.time()
        if current_time - last_capture_time >= 5:
            # Generate a filename with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_dir}/image_{timestamp}.jpg"
            
            # Save the captured frame
            cv2.imwrite(filename, frame)
            print(f"Image saved: {filename}")
            
            # Update the last capture time
            last_capture_time = current_time
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Image capture stopped.")

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
