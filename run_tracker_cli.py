#!/usr/bin/env python3
"""
Minimal CLI runner: reads a video, runs YOLO person detection + PlayerTracker,
writes runs/json/tracks.json and (optionally) runs/videos/out_ids.mp4

Usage:
  python3 run_tracker_cli.py --input /path/to/video.mp4 [--render]

Outputs:
  runs/json/tracks.json
  runs/videos/out_ids.mp4 (only if --render)
"""

import argparse, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np

# Ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    print(f"[ERR] Ultralytics not installed? {e}", file=sys.stderr)
    sys.exit(1)

# Use your existing PlayerTracker class
from tracker_players import PlayerTracker

def ensure_dirs():
    (Path("runs/json")).mkdir(parents=True, exist_ok=True)
    (Path("runs/videos")).mkdir(parents=True, exist_ok=True)

def draw_box(img, x1, y1, x2, y2, tid):
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, f"ID {tid}", (x1, max(0, y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input video")
    ap.add_argument("--render", action="store_true", help="Save overlay video out_ids.mp4")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO model name or path")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO conf threshold")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"[ERR] Input not found: {src}", file=sys.stderr)
        sys.exit(2)

    ensure_dirs()

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[ERR] Failed to open video: {src}", file=sys.stderr)
        sys.exit(3)

    # Load YOLO
    model = YOLO(args.model)

    # Our tracker (people only, class 0)
    tracker = PlayerTracker(conf_thresh=0.20, person_ids=(0,))

    # Prepare writer if rendering
    writer = None
    out_path = Path("runs/videos/out_ids.mp4")
    if args.render:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    tracks_rows = []  # one row per tracked box per frame
    t = 0
    last_log = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # YOLO inference
            res = model.predict(frame, imgsz=640, conf=args.conf, verbose=False)[0]

            # Gather detections
            xyxy = []
            conf = []
            cls  = []
            if res.boxes is not None and len(res.boxes) > 0:
                # Ultralytics tensors → numpy
                b = res.boxes
                xyxy = b.xyxy.cpu().numpy()
                conf = b.conf.cpu().numpy()
                cls  = b.cls.cpu().numpy().astype(np.int32)

            det = {"xyxy": xyxy, "conf": conf, "cls": cls}

            # Update tracker
            tr = tracker.update(det)

            # Collect rows
            for r in tr:
                tracks_rows.append({
                    "t": int(t),
                    "id": int(r["id"]),
                    "x1": float(r["x1"]),
                    "y1": float(r["y1"]),
                    "x2": float(r["x2"]),
                    "y2": float(r["y2"]),
                    "conf": float(r.get("conf", 0.0)),
                    "cls": int(r.get("cls", 0))
                })

            # Optional render
            if writer is not None:
                for r in tr:
                    draw_box(frame, r["x1"], r["y1"], r["x2"], r["y2"], r["id"])
                writer.write(frame)

            t += 1
            if time.time() - last_log > 3:
                print(f"[INFO] frame {t}, tracks {len(tr)}")
                last_log = time.time()

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    # Write JSON
    json_out = Path("runs/json/tracks.json")
    with open(json_out, "w") as f:
        json.dump(tracks_rows, f)

    print(f"[OK] Wrote {json_out} ({len(tracks_rows)} rows)")
    if args.render:
        print(f"[OK] Wrote {out_path}")

if __name__ == "__main__":
    main()
