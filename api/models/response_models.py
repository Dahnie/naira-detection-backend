"""
Pydantic models for API responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

class BoundingBox(BaseModel):
    """Bounding box coordinates model"""
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(BaseModel):
    """Individual detection model"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")

class TopDetection(BaseModel):
    """Top detection model for audio feedback"""
    denomination: str
    confidence: float

class DetectionResponse(BaseModel):
    """Detection API response model"""
    success: bool
    inference_time: float
    detections: List[Detection]
    detection_count: int
    annotated_image: Optional[str] = None
    top_detection: Optional[TopDetection] = None
    error: Optional[str] = None

class AudioResponse(BaseModel):
    """Audio response model"""
    text: str
    type: str = Field(..., description="Type of response (error, no_detection, high_confidence, etc.)")

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    version: str
    model_info: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str