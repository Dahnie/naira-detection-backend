"""
Main FastAPI application for Naira Detection API
"""

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from pydantic import BaseModel
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import routers
from routers import detection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Naira Detection API",
    description="API for detecting Naira notes to assist visually impaired individuals",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

# Templates
templates = Jinja2Templates(directory="webapp/templates")

# Include routers
app.include_router(detection.router, prefix="/api/detection", tags=["Detection"])

# Root endpoint
@app.get("/", tags=["Root"])
async def root(request: Request):
    """Serve the web application interface"""
    # return templates.TemplateResponse("index.html", {"request": request})
    return "Hello world"

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"},
    )

if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)