from ultralytics import YOLO
import cv2
import sys

# Usage: python tools/debug_yolo.py tmp_jobs/VIDEO.mp4
video_path = sys.argv[1]

model = YOLO("yolov8x.pt")
cap = cv2.VideoCapture(video_path)

print(f"🔎 Scanning {video_path} for ANY ball-like objects...")

for i in range(100): # Check first 100 frames
    ret, frame = cap.read()
    if not ret: break
    
    # Extremely low confidence
    results = model.predict(frame, conf=0.01, imgsz=1280, verbose=False)[0]
    
    # Check for Class 32 (Sports Ball)
    balls = [b for b in results.boxes.data.tolist() if int(b[5]) == 32]
    
    if balls:
        print(f"   Frame {i}: Found {len(balls)} candidates. Max Conf: {max([b[4] for b in balls]):.3f}")
    else:
        # What DID it see?
        print(f"   Frame {i}: No balls. Top detection: Class {int(results.boxes.cls[0])} ({results.boxes.conf[0]:.2f})")

cap.release()
