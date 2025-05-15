"""
Detection router for Naira Detection API
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import base64
from pydantic import BaseModel
from typing import Optional, List
import sys
import os
from pathlib import Path
import time
import logging

# Add src directory to path for importing the prediction module
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.predict import NairaDetector

# Configure logging
logger = logging.getLogger(__name__)

# Load detector
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
DATA_YAML = os.environ.get("DATA_YAML", "data/processed/data.yaml")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.5"))

# Initialize the detector
detector = NairaDetector(
    model_path=MODEL_PATH,
    data_yaml=DATA_YAML,
    conf_threshold=CONF_THRESHOLD
)

# Define router
router = APIRouter()

# Request models
class Base64ImageRequest(BaseModel):
    """Request model for base64 encoded image"""
    image: str
    confidence_threshold: Optional[float] = 0.5

class DetectionResult(BaseModel):
    """Response model for detection result"""
    success: bool
    inference_time: float
    detections: List[dict]
    detection_count: int
    annotated_image: Optional[str] = None
    top_detection: Optional[dict] = None
    error: Optional[str] = None

# Endpoints
@router.post("/detect/image", response_model=DetectionResult)
async def detect_from_image(file: UploadFile = File(...), confidence_threshold: Optional[float] = Form(0.5)):
    print("file", file)
    """
    Detect Naira notes from an uploaded image file
    
    - **file**: Image file to analyze
    - **confidence_threshold**: Minimum confidence score for detections (0-1)
    """
    print("file", file)
    try:
        # Check file size (limit to 10MB)
        file_size = 0
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
        # Check file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="File must be an image")
        
        # Encode image to base64 for processing
        img_base64 = base64.b64encode(file_content).decode("utf-8")
        
        # Set confidence threshold
        detector.conf_threshold = confidence_threshold
        
        # Process the image
        result = detector.detect_from_base64(img_base64)
        
        # Log the detection
        logger.info(f"Detection completed: found {result.get('detection_count', 0)} notes")
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.post("/detect/base64", response_model=DetectionResult)
async def detect_from_base64(request: Base64ImageRequest):
    """
    Detect Naira notes from a base64 encoded image
    
    - **image**: Base64 encoded image (with or without data URL prefix)
    - **confidence_threshold**: Minimum confidence score for detections (0-1)
    """
    try:
        # Set confidence threshold
        detector.conf_threshold = request.confidence_threshold
        
        # Process the image
        result = detector.detect_from_base64(request.image)
        
        # Log the detection
        logger.info(f"Detection completed: found {result.get('detection_count', 0)} notes")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing base64 image: {str(e)}")

@router.get("/test", response_model=dict)
async def test_detection():
    """
    Test endpoint to verify the detection service is working
    Returns a simple test result with model information
    """
    try:
        # Get class names
        class_names = list(detector.class_names.values())
        
        return {
            "status": "operational",
            "model_path": MODEL_PATH,
            "classes": class_names,
            "confidence_threshold": detector.conf_threshold
        }
    
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in test endpoint: {str(e)}")