# GPU-enabled base image for RunPod serverless
FROM runpod/serverless:gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs for OpenCV / ffmpeg and some deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Install CUDA 12.1 PyTorch wheels FIRST ----
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# ---- Python deps (your cleaned requirements.txt) ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- Copy all app code (not only handler.py) ----
COPY . /app

# Warm YOLO weights cache (optional)
RUN python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n.pt')
PY

# Serverless entrypoint
CMD ["python", "-u", "handler.py"]
