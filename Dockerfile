# Use RunPod serverless base image with CUDA 12.1
FROM runpod/serverless:gpu-cuda12.1.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install required system libraries for OpenCV, ffmpeg, YOLO, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all source code
COPY . /app

# (Optional) Preload YOLO weights to speed up cold starts
RUN python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n.pt')
PY

# IMPORTANT: Start RunPod worker pointing to your handler
# Your handler.py defines a function `handler(event)`, so the string is `handler.handler`
CMD ["python", "-m", "runpod_serverless.worker", "--handler", "handler.handler"]
