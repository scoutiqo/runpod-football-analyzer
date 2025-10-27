FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# (optional) warm YOLO weights
RUN python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n.pt')
PY

ENV PYTHONUNBUFFERED=1
ENV RUNPOD_DEBUG_LEVEL=DEBUG
ENTRYPOINT ["python", "-m", "runpod.serverless.worker", "--handler-path", "handler.py"]
