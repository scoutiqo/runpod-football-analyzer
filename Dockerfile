# RunPod Serverless base with CUDA 12.1
FROM runpod/serverless:gpu-cuda12.1.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs for OpenCV/FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY . /app

# (Optional) Warm YOLO weights to speed first inference
RUN python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n.pt')
PY

# Start the RunPod Serverless worker; point to your handler function
# If your file is handler.py and the function is run(job): use handler.run
CMD ["python","-m","runpod_serverless.worker","--handler","handler.run"]
