import os
from PIL import Image
from torchvision import transforms
import random

# Set up paths
input_folder = r"good_input_data"  # Path to your original images
output_folder = "output_data"      # Folder to save augmented images

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define transformations
base_transform = transforms.Compose([
    transforms.RandomRotation(degrees=45),  # Rotate ±45°
    transforms.RandomHorizontalFlip(p=0.5), # Horizontal flip
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.2)), # Random zoom
    transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.2, hue=0.1), # Color jitter
])

# Function to apply transformations and save augmented images
def augment_and_save(image, image_name, num_augmentations):
    for i in range(num_augmentations):
        transformed_image = base_transform(image)  # Apply transformations
        transformed_image = transforms.ToPILImage()(transformed_image)  # Convert tensor back to PIL image
        save_path = os.path.join(output_folder, f"{image_name}_aug{i}.jpg")
        transformed_image.save(save_path)
        print(f"Saved {save_path}")

# Number of augmentations to generate per image
num_augmentations = 5

# Loop through each image in the input folder and apply augmentations
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            img = transforms.ToTensor()(img)  # Convert to tensor for compatibility with PyTorch transforms
            augment_and_save(img, filename.split('.')[0], num_augmentations)  # Augment and save