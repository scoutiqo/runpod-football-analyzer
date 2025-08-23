# Minimal image that runs our serverless handler
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# basic libs for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy handler
COPY handler.py /app/handler.py

# serverless entrypoint
CMD ["python", "-u", "handler.py"]
