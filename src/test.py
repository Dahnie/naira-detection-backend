"""
Testing script for Naira Detection Model using YOLOv8
"""

import os
import argparse
from ultralytics import YOLO
import yaml
import cv2
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model for Naira detection")
    parser.add_argument("--data", type=str, default="data/processed/data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to trained weights")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", type=str, default="", help="Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)")
    parser.add_argument("--save-txt", action="store_true", help="Save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="Save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped prediction boxes")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames")
    return parser.parse_args()

def test():
    """Test the trained YOLOv8 model"""
    args = parse_args()
    
    # Load data configuration
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Load model
    print(f"Loading model from {args.weights}")
    model = YOLO(args.weights)
    
    # Run validation on the test set
    print("Running validation on test set...")
    metrics = model.val(
        data=args.data,
        imgsz=args.img_size,
        batch=16,
        device=args.device,
        conf=args.conf_thres,
        iou=args.iou_thres,
        verbose=True
    )
    
    print("\nValidation Results:")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")
    
    # Test on some example images
    print("\nTesting on example images...")
    test_dir = Path(data_config['path']) / data_config['test']
    
    # Get class names
    class_names = data_config['names']
    
    # Create output directory for results
    os.makedirs("results", exist_ok=True)
    
    # Test on a few images from the test set
    for i, img_path in enumerate(list(test_dir.glob("*.jpg"))[:5]):
        print(f"Processing {img_path}...")
        
        # Run inference
        results = model.predict(
            img_path,
            conf=args.conf_thres,
            iou=args.iou_thres,
            imgsz=args.img_size,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            project="results",
            name="test_images",
            exist_ok=True
        )
        
        # Get the image with annotations
        if args.save_frames:
            annotated_img = results[0].plot()
            cv2.imwrite(f"results/test_image_{i}.jpg", annotated_img)
        
        # Print detections
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"Detected {class_names[cls]} with confidence {conf:.4f}")
    
    print("\nTesting completed!")

if __name__ == "__main__":
    test()