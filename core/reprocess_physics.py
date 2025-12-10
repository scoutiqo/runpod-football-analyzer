import json
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# CONFIG
TRACKS_FILE = "runs/json/tracks.json"
VIDEO_DIR = "tmp_jobs"
CALIB_MODEL = "models/pitch_calibration_v1.pt"

# 1. EXACT MATCH with Training Order (Critical)
KEYPOINT_NAMES = [
    "TL_Corner", "TR_Corner", "BR_Corner", "BL_Corner",
    "Center_Circle_Top", "Center_Circle_Bottom", "Center_Spot",
    "Penalty_Spot_Left", "Penalty_Spot_Right",
    "Box_TL_Left", "Box_BL_Left", "Box_TR_Right", "Box_BR_Right"
]

# 2. Real World Coordinates (Meters)
REAL_WORLD_POINTS = {
    "TL_Corner": [0, 0],
    "TR_Corner": [105, 0],
    "BR_Corner": [105, 68],
    "BL_Corner": [0, 68],
    "Center_Circle_Top": [52.5, 34 - 9.15],
    "Center_Circle_Bottom": [52.5, 34 + 9.15],
    "Center_Spot": [52.5, 34],
    "Penalty_Spot_Left": [11, 34],
    "Penalty_Spot_Right": [105 - 11, 34],
    "Box_TL_Left": [16.5, 13.84],
    "Box_BL_Left": [16.5, 54.16],
    "Box_TR_Right": [105 - 16.5, 13.84],
    "Box_BR_Right": [105 - 16.5, 54.16]
}

def main():
    print("📐 INJECTING PHYSICS (FIXED MAPPING)...")
    
    if not Path(TRACKS_FILE).exists(): 
        print("❌ No tracks.json found."); return
        
    data = json.loads(Path(TRACKS_FILE).read_text())
    frames = data.get('frames', [])
    if not frames: return

    # Find video
    videos = list(Path(VIDEO_DIR).glob("*.mp4"))
    if not videos: print("❌ No video found."); return
    video_path = str(max(videos, key=lambda p: p.stat().st_mtime))
    print(f"   🎥 Using video: {video_path}")
    
    model = YOLO(CALIB_MODEL)
    cap = cv2.VideoCapture(video_path)
    
    fixed_count = 0
    
    for i, f_data in enumerate(tqdm(frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, img = cap.read()
        if not ret: break
        
        # Inference
        results = model(img, verbose=False)[0]
        if not results.keypoints: continue
        
        kpts = results.keypoints.xy[0].cpu().numpy()
        confs = results.keypoints.conf[0].cpu().numpy() if results.keypoints.conf is not None else [1.0]*len(kpts)
        
        src_pts = []
        dst_pts = []
        
        # Match Points using Fixed List
        for k_idx, (x, y) in enumerate(kpts):
            if k_idx >= len(KEYPOINT_NAMES): break # Safety
            
            if confs[k_idx] > 0.5: # Threshold
                name = KEYPOINT_NAMES[k_idx]
                if name in REAL_WORLD_POINTS:
                    src_pts.append([x, y])
                    dst_pts.append(REAL_WORLD_POINTS[name])
        
        # Need at least 4 points for Homography
        if len(src_pts) >= 4:
            try:
                H, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts))
                
                # Apply to Players
                for p in f_data.get('players', []):
                    px, py = p['x'] * img.shape[1], p['y'] * img.shape[0]
                    pt = np.array([[[px, py]]], dtype=np.float32)
                    dst = cv2.perspectiveTransform(pt, H)
                    p['x_m'] = float(dst[0][0][0])
                    p['y_m'] = float(dst[0][0][1])
                
                # Apply to Ball
                if f_data.get('ball'):
                    b = f_data['ball']
                    bx, by = b['x'] * img.shape[1], b['y'] * img.shape[0]
                    pt = np.array([[[bx, by]]], dtype=np.float32)
                    dst = cv2.perspectiveTransform(pt, H)
                    b['x_m'] = float(dst[0][0][0])
                    b['y_m'] = float(dst[0][0][1])
                
                fixed_count += 1
            except: pass

    Path(TRACKS_FILE).write_text(json.dumps(data))
    print(f"✅ Injected Physics into {fixed_count} frames.")
    print("   👉 Now you can run: python core/graph_converter.py")

if __name__ == "__main__":
    main()
