import cv2
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Add root to path
sys.path.append(os.getcwd())
from core.camera_movement_estimator import CameraMovementEstimator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="viewer/test_short.mp4")
    parser.add_argument("--out", default="runs/videos/camera_motion.mp4")
    args = parser.parse_args()

    print(f"🎥 Analyzing Camera Motion in {args.video}...")

    cap = cv2.VideoCapture(args.video)
    frames = []
    
    # Read first 150 frames (approx 6 seconds)
    for _ in range(150):
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    if not frames:
        print("❌ Could not read video")
        return

    # 1. Run Estimator
    estimator = CameraMovementEstimator(frames[0])
    camera_moves = estimator.get_camera_movement(frames)
    
    # 2. Render Output
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
    
    # Cumulative movement (to show total pan)
    total_pan_x = 0
    total_pan_y = 0

    for i, frame in enumerate(frames):
        move = camera_moves[i]
        dx, dy = move[0], move[1]
        
        total_pan_x += dx
        total_pan_y += dy
        
        # Draw "Safe Zones" (Where we looked for features)
        # Top/Bottom strips
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, int(h*0.1)), (0, 255, 0), 2) # Top
        cv2.rectangle(overlay, (0, int(h*0.9)), (w, h), (0, 255, 0), 2) # Bottom
        
        # Display Stats
        cv2.putText(frame, f"Pan X: {dx:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Total Pan: {total_pan_x:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        out.write(frame)

    out.release()
    print(f"✅ Camera Motion Video saved to {args.out}")

if __name__ == "__main__":
    main()
