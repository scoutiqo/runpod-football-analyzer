import json
import numpy as np
import cv2
import argparse
from pathlib import Path

# CONFIG
INPUT_TRACKS = "runs/json/formatted_tracks_silver.json"
OUTPUT_TRACKS = "runs/json/formatted_tracks_silver.json" # Enrich in-place

# PITCH DIMENSIONS (Standard FIFA)
PITCH_W = 105.0
PITCH_H = 68.0

# SMOOTHING WINDOW (From your screenshot)
# Calculate speed over 5 frames (0.2s) to ignore micro-jitters
SPEED_WINDOW = 5 

def get_perspective_transform(mask_str, width=1920, height=1080):
    # Source Points (User's manual mask from frontend)
    raw_coords = [float(x) for x in mask_str.split(',')]
    
    # Ensure we have 4 points (x,y * 4)
    if len(raw_coords) < 8:
        # Fallback: Use full screen if mask is bad (Results will be inaccurate but code won't crash)
        raw_coords = [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]

    src_pts = np.array(raw_coords).reshape(-1, 2)
    
    # Denormalize (0-1 -> Pixels)
    src_pts[:, 0] *= width
    src_pts[:, 1] *= height
    src_pts = src_pts.astype(np.float32)

    # Destination Points (Flat 2D Pitch)
    # Order: TL, TR, BR, BL
    dst_pts = np.array([
        [0, 0],            # Top-Left (0,0)
        [PITCH_W, 0],      # Top-Right (105,0)
        [PITCH_W, PITCH_H],# Bottom-Right (105,68)
        [0, PITCH_H]       # Bottom-Left (0,68)
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(src_pts, dst_pts)

def measure_distance(p1, p2):
    # Euclidean distance in Meters
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True, help="Pitch mask: x1,y1,x2,y2,x3,y3,x4,y4")
    args = parser.parse_args()

    print("🏃 Starting SPEED & DISTANCE Estimation (Metric Learning)...")

    if not Path(INPUT_TRACKS).exists():
        print("❌ Tracks not found.")
        return

    data = json.loads(Path(INPUT_TRACKS).read_text())
    frames = data.get('frames', [])
    fps = data.get('fps', 25.0)
    
    # 1. Get Transformation Matrix
    # We assume standard HD reference for the normalized mask
    M = get_perspective_transform(args.mask)

    # 2. Organize Data by Player ID
    # { '1': { 0: [x_m, y_m], 1: [x_m, y_m] ... } }
    player_positions = {}
    
    # First Pass: Calculate Meter Coordinates
    for i, f in enumerate(frames):
        for p in f.get('players', []):
            pid = str(p['id'])
            if pid not in player_positions: player_positions[pid] = {}
            
            # Get Pixels
            px = p['x'] if p['x'] > 1.0 else p['x'] * 1920
            py = p['y'] if p['y'] > 1.0 else p['y'] * 1080
            
            # Transform (Pixels -> Meters)
            pt = np.array([[[px, py]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(pt, M)[0][0]
            
            # Inject into JSON immediately
            p['x_m'] = float(warped[0])
            p['y_m'] = float(warped[1])
            
            player_positions[pid][i] = [warped[0], warped[1]]
            
    # 3. Calculate Speed/Distance
    print(f"   Calculating physics for {len(player_positions)} players...")
    
    for pid, history in player_positions.items():
        total_dist = 0.0
        
        sorted_frames = sorted(history.keys())
        
        for i in range(len(sorted_frames)):
            curr_f = sorted_frames[i]
            
            # SPEED (Look back 'SPEED_WINDOW' frames)
            speed_kmh = 0.0
            if i >= SPEED_WINDOW:
                prev_f = sorted_frames[i - SPEED_WINDOW]
                
                # Only calculate if frames are contiguous in time
                if (curr_f - prev_f) == SPEED_WINDOW:
                    p1 = history[prev_f]
                    p2 = history[curr_f]
                    
                    dist_m = measure_distance(p1, p2)
                    time_s = SPEED_WINDOW / fps
                    
                    speed_mps = dist_m / time_s
                    speed_kmh = speed_mps * 3.6
                    
                    # Human limit cap (Usain Bolt ~44 km/h)
                    if speed_kmh > 45: speed_kmh = 0 

            # DISTANCE (Accumulative)
            if i > 0:
                prev_f = sorted_frames[i-1]
                if (curr_f - prev_f) == 1:
                    p1 = history[prev_f]
                    p2 = history[curr_f]
                    dist = measure_distance(p1, p2)
                    if dist < 2.0: # Ignore teleportation glitches
                        total_dist += dist
                        
            # Inject into the main JSON structure
            # (We have to find the frame again)
            for p in frames[curr_f]['players']:
                if str(p['id']) == pid:
                    p['speed'] = round(speed_kmh, 1)
                    p['dist'] = round(total_dist / 1000, 3) # km
                    break

    # 4. Save Enriched Tracks
    Path(OUTPUT_TRACKS).write_text(json.dumps(data))
    print(f"✅ Added Physics Metrics to {OUTPUT_TRACKS}")

if __name__ == '__main__':
    main()
