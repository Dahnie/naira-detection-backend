# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent tzdata prompts and set locale
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Update package list and install system dependencies in separate steps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libgl1-mesa-glx \
        libgthread-2.0-0 \
        libfontconfig1 \
        libxrender1 \
        libxtst6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install ffmpeg separately (sometimes causes issues)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

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