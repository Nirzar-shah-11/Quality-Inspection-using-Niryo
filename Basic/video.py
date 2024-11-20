import cv2
print("hello")
# Load the video capture (0 for default webcam, 1 if using an external camera)
videoCap = cv2.VideoCapture(0)

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue

    # Show the video frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()