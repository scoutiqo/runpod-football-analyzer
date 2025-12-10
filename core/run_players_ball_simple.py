#!/usr/bin/env python

import argparse
import json
import cv2
from ultralytics import YOLO
import numpy as np
import os

"""
GENERALIZED VERSION
Allows:
    python core/run_players_ball_simple.py --video INPUT.mp4 --out OUTPUT.json
"""

def main():
    parser = argparse.ArgumentParser(description="Player + Ball Tracker (Simple)")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", required=True, help="Where to write tracks JSON")

    args = parser.parse_args()
    video_path = args.video
    out_path = args.out

    print(f"Loading model: yolov8n.pt")
    model = YOLO("yolov8n.pt")  # uses repo-local weights

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model.predict(frame, imgsz=640, conf=0.15, verbose=False)[0]

        dets = []
        for box in res.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            dets.append({
                "cls": cls,
                "bbox": [x1, y1, x2, y2]
            })

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}, detections: {len(dets)}")

        results.append({
            "frame": frame_idx,
            "detections": dets
        })

        frame_idx += 1

    cap.release()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "fps": fps,
            "frames": results
        }, f)

    print(f"Wrote {out_path} with {len(results)} frames")

if __name__ == "__main__":
    main()
