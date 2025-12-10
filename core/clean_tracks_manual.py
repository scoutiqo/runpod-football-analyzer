#!/usr/bin/env python
import json
import numpy as np
import cv2
import argparse
import os
from pathlib import Path

# INPUT / OUTPUT
INPUT_TRACKS = "runs/json/formatted_tracks_silver.json"
OUTPUT_TRACKS = "runs/json/formatted_tracks_silver.json"

# TRAINING DATA STORAGE
DATASET_DIR = Path("datasets/pitch_segmentation")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def save_training_data(video_path, poly_pts):
    """
    Saves the raw frame and the user-defined mask as a training pair.
    This builds your proprietary dataset automatically.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret: return

        # Create unique filename based on video hash or name
        base_name = Path(video_path).stem
        img_path = DATASET_DIR / f"{base_name}_img.jpg"
        mask_path = DATASET_DIR / f"{base_name}_mask.png"
        
        # 1. Save Image
        cv2.imwrite(str(img_path), frame)
        
        # 2. Create and Save Mask (White polygon on Black background)
        h, w = frame.shape[:2]
        mask_img = np.zeros((h, w), dtype=np.uint8)
        
        # Denormalize points
        poly_px = (poly_pts * [w, h]).astype(np.int32)
        cv2.fillPoly(mask_img, [poly_px], 255)
        
        cv2.imwrite(str(mask_path), mask_img)
        
        print(f"   💾 SAVED TRAINING DATA: {img_path}")
        print(f"      (Your manual effort is now part of the AI's memory)")
        
    except Exception as e:
        print(f"   ⚠️ Failed to save training data: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True, help="Comma-separated polygon coords")
    # We need the video path to take the screenshot
    parser.add_argument("--video", default="viewer/test_short.mp4", help="Path to source video")
    args = parser.parse_args()

    print("🧹 Starting MANUAL CLEANING + LEARNING LOOP...")

    # 1. Parse Mask
    try:
        coords = [float(x) for x in args.mask.split(',')]
        poly_pts = np.array(coords).reshape(-1, 2).astype(np.float32)
    except Exception:
        return

    # 2. SAVE THE KNOWLEDGE (The Learning Step)
    save_training_data(args.video, poly_pts)

    # 3. CLEAN THE TRACKS (The Processing Step)
    if not Path(INPUT_TRACKS).exists(): return
    tracks_data = json.loads(Path(INPUT_TRACKS).read_text())
    
    REF_W, REF_H = 1920.0, 1080.0 
    cleaned_frames = []
    
    for f in tracks_data.get('frames', []):
        valid_players = []
        for p in f.get('players', []):
            # Coordinate logic...
            if 'x' in p and p['x'] <= 1.0: nx, ny = p['x'], p['y']
            elif 'x_px' in p: nx, ny = p['x_px'] / REF_W, p['y_px'] / REF_H
            else: continue

            dist = cv2.pointPolygonTest(poly_pts, (nx, ny), False)
            if dist >= 0: valid_players.append(p)
        
        ball = f.get('ball')
        if ball:
             # Ball logic...
             if 'x' in ball and ball['x'] <= 1.0: bx, by = ball['x'], ball['y']
             else: bx, by = ball['x_px'] / REF_W, ball['y_px'] / REF_H
             if cv2.pointPolygonTest(poly_pts, (bx, by), False) < 0: ball = None
        
        cleaned_frames.append({"t": f['t'], "ball": ball, "players": valid_players})

    tracks_data['frames'] = cleaned_frames
    Path(OUTPUT_TRACKS).write_text(json.dumps(tracks_data))
    print(f"✅ Saved masked tracks.")

if __name__ == '__main__':
    main()
