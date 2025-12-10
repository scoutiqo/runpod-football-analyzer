#!/usr/bin/env python
"""
debug_yolo_detections.py

Run pure YOLO detections (no tracking), filter to "person" only,
draw boxes + counts, and write a debug video.

Usage:
  python core/debug_yolo_detections.py \
    --input uploads/test_short.mp4 \
    --model yolov8m.pt \
    --conf 0.65 \
    --save runs/videos/debug_yolo_m.mp4
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input video")
    ap.add_argument("--model", required=True, help="YOLO model name or path")
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    ap.add_argument("--save", required=True, help="Output debug video path")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input video not found: {in_path}")

    print(f"[debug_yolo] Loading model: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Run YOLO on this frame
        results = model(frame, conf=args.conf, verbose=False)[0]
        boxes = results.boxes

        person_count = 0

        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls = int(b.cls.item())
                # COCO "person" == class 0. Adjust if your model differs.
                if cls != 0:
                    continue

                person_count += 1
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf.item())

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"p {conf:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        # Overlay count
        cv2.putText(
            frame,
            f"frame {frame_idx} | persons: {person_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if frame_idx % 25 == 0:
            print(f"[debug_yolo] frame {frame_idx}: person_dets={person_count}")

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[debug_yolo] Done. Wrote {out_path} with {frame_idx} frames.")


if __name__ == "__main__":
    main()
