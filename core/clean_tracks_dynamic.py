#!/usr/bin/env python
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# CONFIG
INPUT_TRACKS = "runs/json/formatted_tracks_silver.json"
VIDEO_SOURCE = "viewer/test_short.mp4"
OUTPUT_TRACKS = "runs/json/formatted_tracks_silver.json"

def get_adaptive_pitch_mask(frame):
    """
    Dynamically finds the pitch contour for THIS specific frame.
    1. Samples the center.
    2. thresholds based on sample stats.
    3. Returns the Convex Hull of the largest blob.
    """
    h, w = frame.shape[:2]
    
    # 1. Sample the Center (20% box) - The "Safe Zone"
    center_h, center_w = int(h/2), int(w/2)
    sample_h, sample_w = int(h*0.1), int(w*0.1)
    
    sample_patch = frame[center_h-sample_h:center_h+sample_h, 
                         center_w-sample_w:center_w+sample_w]
    
    # Convert to HSV for robust color stats
    hsv_sample = cv2.cvtColor(sample_patch, cv2.COLOR_BGR2HSV)
    
    # Calculate mean and std dev of the grass color
    mean, std = cv2.meanStdDev(hsv_sample)
    
    # Define dynamic range (Mean +/- 2.5 StdDevs) - Captures shadows/highlights
    # We widen the Hue tolerance because grass hue varies less than Lightness
    lower = np.array([max(0, mean[0][0] - 20), 
                      max(0, mean[1][0] - 40), 
                      max(0, mean[2][0] - 60)])
                      
    upper = np.array([min(180, mean[0][0] + 20), 
                      min(255, mean[1][0] + 40), 
                      min(255, mean[2][0] + 60)])
    
    # 2. Create Mask
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower, upper)
    
    # 3. Clean Noise (Morphology)
    # Close small holes (players)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Open to remove small isolated blobs (noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 4. Find Largest Contour (The Pitch)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts: return None
    
    largest_cnt = max(cnts, key=cv2.contourArea)
    
    # 5. Convex Hull (Smooths edges, ignores benches sticking in)
    hull = cv2.convexHull(largest_cnt)
    
    return hull

def main():
    print("🧹 Starting DYNAMIC ADAPTIVE CLEANING...")
    
    if not Path(INPUT_TRACKS).exists():
        print("❌ Tracks not found.")
        return

    tracks_data = json.loads(Path(INPUT_TRACKS).read_text())
    frames = tracks_data.get('frames', [])
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cleaned_frames = []
    removed_count = 0
    
    # We process frame by frame because the camera might move
    for i in tqdm(range(total_frames)):
        ret, img = cap.read()
        if not ret: break
        
        if i >= len(frames): break
        f_data = frames[i]
        
        # Get the Pitch Polygon for THIS frame
        pitch_hull = get_adaptive_pitch_mask(img)
        
        valid_players = []
        
        if pitch_hull is not None:
            for p in f_data.get('players', []):
                # Get coordinates
                if 'x_px' in p: x, y = int(p['x_px']), int(p['y_px'])
                elif p['x'] > 1.0: x, y = int(p['x']), int(p['y'])
                else: x, y = int(p['x'] * img.shape[1]), int(p['y'] * img.shape[0])
                
                # Point Polygon Test
                # measure distance to polygon. +ve is inside, -ve is outside
                # 5.0 pixel buffer allowed
                dist = cv2.pointPolygonTest(pitch_hull, (x, y), True)
                
                if dist >= -10.0: # Allow being slightly on the line
                    valid_players.append(p)
                else:
                    removed_count += 1
        else:
            # Fallback: If mask fails (rare), keep all players to be safe
            valid_players = f_data.get('players', [])

        # Filter Ball as well
        ball = f_data.get('ball')
        if ball and pitch_hull is not None:
             if 'x_px' in ball: bx, by = int(ball['x_px']), int(ball['y_px'])
             elif ball['x'] > 1.0: bx, by = int(ball['x']), int(ball['y'])
             else: bx, by = int(ball['x'] * img.shape[1]), int(ball['y'] * img.shape[0])
             
             if cv2.pointPolygonTest(pitch_hull, (bx, by), False) < 0:
                 ball = None # Ball is out of bounds

        cleaned_frames.append({
            "t": f_data['t'],
            "ball": ball,
            "players": valid_players
        })

    cap.release()
    
    print(f"   Removed {removed_count} detections dynamically.")
    
    tracks_data['frames'] = cleaned_frames
    Path(OUTPUT_TRACKS).write_text(json.dumps(tracks_data))
    print(f"✅ Saved adaptive tracks to {OUTPUT_TRACKS}")

if __name__ == '__main__':
    main()
