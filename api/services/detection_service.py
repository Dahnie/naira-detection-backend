"""
Detection service for Naira notes
"""

import sys
import os
from pathlib import Path
import logging
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import time

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.predict import NairaDetector

# Configure logging
logger = logging.getLogger(__name__)

class DetectionService:
    """Service for detecting Naira notes"""
    
    def __init__(self):
        """Initialize the detection service"""
        # Load environment variables
        model_path = os.environ.get("MODEL_PATH", "models/best.pt")
        data_yaml = os.environ.get("DATA_YAML", "data/processed/data.yaml")
        conf_threshold = float(os.environ.get("CONF_THRESHOLD", "0.5"))
        
        # Initialize detector
        self.detector = NairaDetector(
            model_path=model_path,
            data_yaml=data_yaml,
            conf_threshold=conf_threshold
        )
        
        logger.info(f"Detection service initialized with model: {model_path}")
    
    async def detect_from_file(self, file_content, confidence_threshold=None):
        """
        Detect Naira notes from file content
        
        Args:
            file_content: Bytes content of the image file
            confidence_threshold: Optional confidence threshold override
            
        Returns:
            Detection result dictionary
        """
        try:
            # Set confidence threshold if provided
            if confidence_threshold is not None:
                self.detector.conf_threshold = confidence_threshold
            
            # Convert to base64
            img_base64 = base64.b64encode(file_content).decode("utf-8")
            
            # Process the image
            result = self.detector.detect_from_base64(img_base64)
            
            return result
        
        except Exception as e:
            logger.error(f"Error in detect_from_file: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "inference_time": 0,
                "detections": [],
                "detection_count": 0
            }
    
    async def detect_from_base64(self, base64_string, confidence_threshold=None):
        """
        Detect Naira notes from base64 encoded image
        
        Args:
            base64_string: Base64 encoded image string
            confidence_threshold: Optional confidence threshold override
            
        Returns:
            Detection result dictionary
        """
        try:
            # Set confidence threshold if provided
            if confidence_threshold is not None:
                self.detector.conf_threshold = confidence_threshold
            
            # Process the image
            result = self.detector.detect_from_base64(base64_string)
            
            return result
        
        except Exception as e:
            logger.error(f"Error in detect_from_base64: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "inference_time": 0,
                "detections": [],
                "detection_count": 0
            }
    
    def get_audio_response(self, detection_result):
        """
        Generate appropriate audio response based on detection result
        
        Args:
            detection_result: Detection result dictionary
            
        Returns:
            Dictionary with audio response information
        """
        if not detection_result.get("success", False):
            return {
                "text": "Sorry, I couldn't process that image. Please try again.",
                "type": "error"
            }
        
        if detection_result.get("detection_count", 0) == 0:
            return {
                "text": "No Naira notes detected in the image. Please try again with a clearer image.",
                "type": "no_detection"
            }
        
        # Get the top detection
        top_detection = detection_result.get("top_detection")
        if top_detection:
            denomination = top_detection.get("denomination", "").replace("-naira", " Naira")
            confidence = top_detection.get("confidence", 0) * 100
            
            # Format the response based on confidence
            if confidence > 90:
                return {
                    "text": f"This is a {denomination} note.",
                    "type": "high_confidence"
                }
            elif confidence > 70:
                return {
                    "text": f"This appears to be a {denomination} note.",
                    "type": "medium_confidence"
                }
            else:
                return {
                    "text": f"This might be a {denomination} note, but I'm not very confident.",
                    "type": "low_confidence"
                }
        
        return {
            "text": "Detection completed, but I couldn't determine the note denomination.",
            "type": "unknown"
        }

# Create a singleton instance
detection_service = DetectionService()