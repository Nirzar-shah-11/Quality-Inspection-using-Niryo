import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split


# Transformations for your dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load dataset
dataset = datasets.ImageFolder(root="/Users/nirzarshah/Documents/train", transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class BottleClassifierCNN(nn.Module):
    def __init__(self):
        super(BottleClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: good and bad

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = BottleClassifierCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(10):  # Adjust epochs as needed
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

# Evaluation function

def evaluate(model, data_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return test_loss / len(data_loader), accuracy

# Training and evaluation loop
num_epochs = 10
for epoch in range(num_epochs):
    test_loss, accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Testing Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

print("Training and evaluation completed.")
model.eval()

# # Transformation for webcam frames
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Preprocess and classify
#     img = transform(frame).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(img)
#         _, pred = torch.max(output, 1)
    
#     label = "Good" if pred.item() == 0 else "Bad"
#     cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Good" else (0, 0, 255), 2)
#     cv2.imshow("Bottle Classification", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# import os
# import torch
# from torch import nn, optim
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import transforms, models
# from PIL import Image
# from tqdm import tqdm

# # Dataset class for Coke can classification
# class CokeCanDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.img_labels = []
#         for label, class_dir in enumerate(['good', 'bad']):
#             class_path = os.path.join(root_dir, class_dir)
#             for img_name in os.listdir(class_path):
#                 img_path = os.path.join(class_dir, img_name)
#                 self.img_labels.append((img_path, label))

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path, label = self.img_labels[idx]
#         img = Image.open(os.path.join(self.root_dir, img_path)).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img, label

# # Set up transformations and dataset
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resizing images for input into pre-trained model
#     transforms.ToTensor()
# ])
# dataset = CokeCanDataset(root_dir='/Users/nirzarshah/Documents/train', transform=transform)

# # Split dataset into training and testing sets
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# # Load pre-trained ResNet model and modify for binary classification
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 2)  # Output layer for 'good' and 'bad' classes

# # Define training components
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)
# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.001)

# # # Training function
# # def train_one_epoch(model, optimizer, data_loader, criterion, device):
# #     model.train()
# #     running_loss = 0.0
# #     for images, labels in tqdm(data_loader, desc="Training"):
# #         images, labels = images.to(device), labels.to(device)
        
# #         optimizer.zero_grad()
# #         outputs = model(images)
# #         loss = criterion(outputs, labels)
# #         loss.backward()
# #         optimizer.step()
        
# #         running_loss += loss.item()
# #     return running_loss / len(data_loader)

# # # Evaluation function
# # def evaluate(model, data_loader, criterion, device):
# #     model.eval()
# #     test_loss = 0.0
# #     correct = 0
# #     total = 0
# #     with torch.no_grad():
# #         for images, labels in tqdm(data_loader, desc="Evaluating"):
# #             images, labels = images.to(device), labels.to(device)
# #             outputs = model(images)
# #             loss = criterion(outputs, labels)
# #             test_loss += loss.item()
            
# #             _, predicted = torch.max(outputs, 1)
# #             correct += (predicted == labels).sum().item()
# #             total += labels.size(0)
# #     accuracy = correct / total
# #     return test_loss / len(data_loader), accuracy

# # # Training and evaluation loop
# # num_epochs = 10
# # for epoch in range(num_epochs):
# #     print(f"Epoch {epoch+1}/{num_epochs}")
# #     train_loss = train_one_epoch(model, optimizer, train_loader, criterion, device)
# #     print(f"Training Loss: {train_loss:.4f}")
# #     test_loss, accuracy = evaluate(model, test_loader, criterion, device)
# #     print(f"Testing Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

# # print("Training and evaluation completed.")
