"""
Training script for Naira Detection Model using YOLOv8
"""

import os
import argparse
from ultralytics import YOLO
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for Naira detection")
    parser.add_argument("--data", type=str, default="data/processed/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="YOLOv8 model size")
    parser.add_argument("--weights", type=str, default=None, help="Path to initial weights")
    parser.add_argument("--device", type=str, default="", help="Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)")
    return parser.parse_args()

def train():
    """Train the YOLOv8 model"""
    args = parse_args()
    
    # Load data configuration
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
        
    print(f"Training with dataset: {data_config}")
    
    # Initialize model
    if args.weights:
        print(f"Loading initial weights from: {args.weights}")
        model = YOLO(args.weights)
    else:
        print(f"Initializing YOLOv8{args.model_size} model with pre-trained weights")
        model = YOLO(f"yolov8{args.model_size}.pt")
    
    # Start training
    print(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        patience=20,  # Early stopping patience
        save=True,    # Save checkpoints
        device=args.device,
        project="models",
        name="naira_detection",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42
    )
    
    print("Training completed!")
    
    # Copy the best model to a standard location
    best_model_path = os.path.join("models", "naira_detection", "weights", "best.pt")
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, os.path.join("models", "best.pt"))
        print(f"Best model saved to models/best.pt")

if __name__ == "__main__":
    train()