import json
import cv2
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# Add project root to path to import the class
sys.path.append(os.getcwd())
from core.camera_movement_estimator import CameraMovementEstimator

# CONFIG
INPUT_TRACKS = "runs/json/formatted_tracks_silver.json" # The clean tracks
OUTPUT_TRACKS = "runs/json/formatted_tracks_silver.json" # Enrich in-place

def main():
    parser = argparse.ArgumentParser()
    # We need the video to calculate optical flow
    # We will try to find the video path, or accept it as an arg
    parser.add_argument("--video", help="Path to video file", default="viewer/test_short.mp4")
    args = parser.parse_args()

    print("🎥 Starting Camera Stabilization (Optical Flow)...")
    
    if not Path(INPUT_TRACKS).exists():
        print("❌ Tracks not found.")
        return
        
    if not os.path.exists(args.video):
        # Fallback: Try to find it in tmp_jobs if not provided
        # This is a hack for the pipeline. Ideally, pipeline passes it.
        print(f"⚠️ Video not found at {args.video}. Stabilization skipped.")
        return

    # 1. Load Video Frames
    cap = cv2.VideoCapture(args.video)
    frames = []
    
    # Read all frames (Memory intensive, but necessary for optical flow batch)
    # Optimisation: We can read frame-by-frame in the estimator, but let's stick to the class structure
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    if not frames:
        print("❌ Failed to read video frames.")
        return

    # 2. Calculate Camera Movement
    estimator = CameraMovementEstimator(frames[0])
    camera_movement = estimator.get_camera_movement(frames)
    
    # 3. Inject into Tracks
    tracks_data = json.loads(Path(INPUT_TRACKS).read_text())
    track_frames = tracks_data.get('frames', [])
    
    print(f"   Injecting camera offsets into {len(track_frames)} frames...")
    
    for i, f in enumerate(track_frames):
        if i >= len(camera_movement): break
        
        move_x, move_y = camera_movement[i]
        
        # Add global offset to the frame object
        f['camera_offset'] = {'x': move_x, 'y': move_y}
        
        # Also add to every player for easy access in speed estimator
        for p in f.get('players', []):
            p['cam_x'] = move_x
            p['cam_y'] = move_y
            
    # 4. Save
    Path(OUTPUT_TRACKS).write_text(json.dumps(tracks_data))
    print("✅ Camera Stabilization complete. Tracks updated.")

if __name__ == "__main__":
    main()
