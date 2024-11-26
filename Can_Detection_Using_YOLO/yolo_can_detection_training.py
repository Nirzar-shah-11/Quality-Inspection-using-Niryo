from ultralytics import YOLO

# Load the pre-trained YOLOv8 model (you can use yolov8s.pt, yolov8m.pt, or yolov8l.pt)
model = YOLO("yolov8s.pt")

# Train the model on your custom dataset
results = model.train(data="/Users/nirzarshah/Documents/Pyniryo/Dataset_creation/config.yaml", epochs=20, imgsz=640)

# Print the results after training
print(results)

# Save the trained model
model.save("yolov8_custom.pt")
