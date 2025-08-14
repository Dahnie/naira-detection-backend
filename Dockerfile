# Build stage
FROM python:3.11 as builder

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Install essential runtime libraries for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libxext6 \
    libsm6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# Set OpenGL environment variables for headless operation
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV MESA_GLSL_VERSION_OVERRIDE=330

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
# Add the app directory to Python path so imports work correctly
ENV PYTHONPATH=/app:/app/api

# Copy the entire application
COPY . .

# Create directories for YOLOv8 models and cache
RUN mkdir -p /app/.cache/ultralytics

# Expose port
EXPOSE 8000

# Environment variables
ENV MODEL_PATH=models/best.pt
ENV DATA_YAML=data/processed/data.yaml
ENV CONF_THRESHOLD=0.5
ENV PORT=8000

# Run the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]