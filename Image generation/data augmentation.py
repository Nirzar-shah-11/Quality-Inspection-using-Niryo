import os
from PIL import Image
from torchvision import transforms
import random

# Set up paths
input_folder = r"D:/Project/.venv/data/New folder"  # Path to your original images
output_folder = "D:/Project/.venv/data/good_aug"              # Folder to save augmented images

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define transformations
augmentation_transforms = [
    transforms.RandomRotation(degrees=45),              # Rotate ±15°
    transforms.RandomHorizontalFlip(p=4),               # Horizontal flip
    transforms.RandomVerticalFlip(p=2),                 # Vertical flip
    transforms.RandomHorizontalFlip(p=3),
    transforms.RandomVerticalFlip(p=1),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.2)), # Random zoom
    transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.2, hue=0.1), # Color jitter
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # Gaussian blur
]

# Function to apply transformations and save augmented images
def augment_and_save(image, image_name):
    for i, transform in enumerate(augmentation_transforms):
        transformed_image = transform(image)  # Apply the transformation
        transformed_image = transforms.ToPILImage()(transformed_image)  # Convert tensor back to PIL image
        save_path = os.path.join(output_folder, f"{image_name}_aug{i}.jpg")
        transformed_image.save(save_path)
        print(f"Saved {save_path}")

# Loop through each image in the input folder and apply augmentations
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            img = transforms.ToTensor()(img)  # Convert to tensor for compatibility with PyTorch transforms
            augment_and_save(img, filename.split('.')[0])  # Augment and save
