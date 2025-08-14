# ===== Stage 1: Builder =====
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system packages required for building wheels (gcc, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install them into a temp directory
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ===== Stage 2: Final runtime image =====
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies needed for OpenCV, Torch, YOLOv8
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy the rest of your application code
COPY . .

# Create cache dir for YOLOv8
RUN mkdir -p /app/.cache/ultralytics

# Expose port
EXPOSE 8000

# Set environment variables
ENV MODEL_PATH=models/best.pt
ENV DATA_YAML=data/processed/data.yaml
ENV CONF_THRESHOLD=0.5
ENV PORT=8000

# Start the app with Gunicorn + Uvicorn workers (production)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api.main:app", "--bind", "0.0.0.0:8000", "--workers", "3"]
