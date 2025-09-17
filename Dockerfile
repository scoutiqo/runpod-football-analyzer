# Use RunPod official GPU image
FROM runpod/serverless:gpu-cuda12.1.1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project code
COPY . .

# Preload YOLO weights (optional, speeds up first run)
RUN python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n.pt')
PY

# IMPORTANT: RunPod serverless entrypoint
CMD ["python", "-m", "runpod.serverless.worker", "--handler", "handler.handler"]
