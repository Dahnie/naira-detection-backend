"""
Prediction utility for Naira Detection Model
"""

import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import time
import argparse

class NairaDetector:
    """Naira note detector class using YOLOv8"""
    
    def __init__(self, model_path="models/best.pt", data_yaml="data/processed/data.yaml", conf_threshold=0.5):
        """
        Initialize the Naira detector
        
        Args:
            model_path: Path to the trained YOLOv8 model
            data_yaml: Path to the data.yaml file containing class names
            conf_threshold: Confidence threshold for detections
        """
        # Load the model
        self.model = YOLO(model_path)
        
        # Load class names from data.yaml
        with open(data_yaml, 'r') as f:
            self.class_names = yaml.safe_load(f)['names']
        
        self.conf_threshold = conf_threshold
        print(f"Naira Detector initialized with {len(self.class_names)} classes")
    
    def detect_from_file(self, image_path):
        """
        Detect naira notes from an image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with detection results
        """
        # Check if file exists
        if not os.path.exists(image_path):
            return {"error": f"File {image_path} not found"}
        
        # Run inference
        start_time = time.time()
        results = self.model.predict(image_path, conf=self.conf_threshold)
        inference_time = time.time() - start_time
        
        # Process results
        return self._process_results(results[0], inference_time)
    
    def detect_from_image(self, image):
        """
        Detect naira notes from an image array
        
        Args:
            image: numpy array image (BGR format from OpenCV)
            
        Returns:
            Dictionary with detection results
        """
        # Run inference
        start_time = time.time()
        results = self.model.predict(image, conf=self.conf_threshold)
        inference_time = time.time() - start_time
        
        # Process results
        return self._process_results(results[0], inference_time)
    
    def detect_from_base64(self, base64_string):
        """
        Detect naira notes from a base64 encoded image
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Decode base64 string
            if "base64," in base64_string:
                base64_string = base64_string.split("base64,")[1]
            
            image_bytes = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to numpy array (RGB to BGR for OpenCV)
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Run detection
            return self.detect_from_image(image_np)
        
        except Exception as e:
            return {"error": f"Error processing base64 image: {str(e)}"}
    
    def _process_results(self, results, inference_time):
        """
        Process YOLOv8 results into a structured format
        
        Args:
            results: YOLOv8 results object
            inference_time: Time taken for inference
            
        Returns:
            Dictionary with processed results
        """
        # Get boxes, confidence scores, and class IDs
        boxes = results.boxes
        
        # Extract detections
        detections = []
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Get class ID and name
            cls_id = int(box.cls[0].item())
            cls_name = self.class_names[cls_id]
            
            # Get confidence score
            conf = float(box.conf[0].item())
            
            # Add to detections list
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
        
        # Sort detections by confidence (highest first)
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Create annotated image with bounding boxes
        annotated_img = results.plot()
        
        # Encode annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response = {
            "success": True,
            "inference_time": inference_time,
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "detection_count": len(detections)
        }
        
        # Find the most confident detection for audio feedback
        if detections:
            top_detection = detections[0]
            response["top_detection"] = {
                "denomination": top_detection["class_name"],
                "confidence": top_detection["confidence"]
            }
        else:
            response["top_detection"] = None
        
        return response

def main():
    """Main function to run predictions on images"""
    parser = argparse.ArgumentParser(description="Run Naira detection on images")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to trained model weights")
    parser.add_argument("--data", type=str, default="data/processed/data.yaml", help="Path to data.yaml")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    # Check if image is provided
    if not args.image:
        print("Please provide an input image with --image")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    detector = NairaDetector(args.model, args.data, args.conf)
    
    # Run detection
    results = detector.detect_from_file(args.image)
    
    # Print results
    print(f"Detection completed in {results['inference_time']:.4f} seconds")
    print(f"Found {results['detection_count']} naira notes")
    
    for i, det in enumerate(results['detections']):
        print(f"{i+1}. {det['class_name']} with confidence {det['confidence']:.4f}")
    
    # Save annotated image
    img_data = results['annotated_image'].split("base64,")[1]
    img_bytes = base64.b64decode(img_data)
    
    output_path = os.path.join(args.output, os.path.basename(args.image))
    with open(output_path, 'wb') as f:
        f.write(img_bytes)
    
    print(f"Annotated image saved to {output_path}")

if __name__ == "__main__":
    main()