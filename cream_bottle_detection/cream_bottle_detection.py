import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('yolov8_custom1.pt')  # Replace with the path to your trained model

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform inference on the webcam frame
    results = model(frame)

    # Display each result in the results list
    for result in results:
        result_img = result.plot()  # Get the image with bounding boxes and labels
        cv2.imshow("Object Detection", result_img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

