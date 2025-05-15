"""
Data preparation script for Naira detection model
- Split dataset into train/validation/test
- Organize into YOLOv8 format
"""

import os
import random
import shutil
from pathlib import Path
import yaml
from PIL import Image
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
SOURCE_DIR = "data/raw"
TARGET_DIR = "data/processed"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
CLASSES = ["5-naira", "10-naira", "20-naira", "50-naira", "100-naira", "200-naira", "500-naira", "1000-naira"]

def create_directory_structure():
    """Create the necessary directory structure for YOLOv8"""
    for split in ["train", "val", "test"]:
        for folder in ["images", "labels"]:
            os.makedirs(os.path.join(TARGET_DIR, split, folder), exist_ok=True)
    
    print("✅ Directory structure created")

def convert_labels(label_path, image_width, image_height, class_id):
    """
    Convert standard bounding box to YOLO format
    YOLO format: [class_id, x_center, y_center, width, height] - all normalized
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    yolo_labels = []
    for line in lines:
        parts = line.strip().split()
        # Assuming the raw format is x_min, y_min, x_max, y_max
        x_min, y_min, x_max, y_max = map(float, parts)
        
        # Convert to YOLO format (normalized)
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
        
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_labels

def process_dataset():
    """Process the raw dataset and organize it into YOLO format"""
    # Get all image files
    all_images = []
    for class_id, class_name in enumerate(CLASSES):
        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist, skipping...")
            continue
        
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                label_path = img_path.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                
                # Check if label exists
                if os.path.exists(label_path):
                    all_images.append((img_path, label_path, class_id))
    
    # Split the dataset
    train_data, temp_data = train_test_split(all_images, test_size=(VAL_RATIO + TEST_RATIO), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO), random_state=42)
    
    print(f"Split dataset: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
    
    # Process and copy the data
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        for img_path, label_path, class_id in split_data:
            # Get image dimensions
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Convert labels to YOLO format
            yolo_labels = convert_labels(label_path, img_width, img_height, class_id)
            
            # Copy the image
            filename = os.path.basename(img_path)
            dst_img_path = os.path.join(TARGET_DIR, split_name, "images", filename)
            shutil.copy(img_path, dst_img_path)
            
            # Save the YOLO format label
            dst_label_path = os.path.join(TARGET_DIR, split_name, "labels", os.path.basename(img_path).replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))
    
    print("✅ Dataset processed and organized in YOLO format")

def create_data_yaml():
    """Create the data.yaml file for YOLOv8"""
    data_yaml = {
        "path": "../data/processed",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(CLASSES)}
    }
    
    with open(os.path.join(TARGET_DIR, "data.yaml"), 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print("✅ data.yaml created")

def main():
    """Main function to run the data preparation process"""
    print("Starting data preparation for Naira detection model...")
    
    # Create directory structure
    create_directory_structure()
    
    # Process the dataset
    process_dataset()
    
    # Create data.yaml
    create_data_yaml()
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()