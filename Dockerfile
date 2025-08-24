# Uses a public image with CUDA 12.1 + PyTorch already installed
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs for OpenCV/ffmpeg and some deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (your requirements INCLUDE the 'runpod' Python package)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy all app code
COPY . /app

# Warm YOLO weights (optional but speeds cold start)
RUN python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n.pt')
PY

# Serverless handler entry
CMD ["python", "-u", "handler.py"]
