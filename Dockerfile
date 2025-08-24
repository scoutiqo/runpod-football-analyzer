FROM runpod/serverless:gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# system libs for cv/ffmpeg and some deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch pinned to CUDA 12.1
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy all app code
COPY . /app

# warm YOLO weights (faster cold starts)
RUN python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n.pt')
PY

CMD ["python", "-u", "handler.py"]
