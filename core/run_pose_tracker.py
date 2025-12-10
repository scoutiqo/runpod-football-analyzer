#!/usr/bin/env python3
import argparse, sys
import numpy as np
import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--conf", type=float, default=0.10)
    args = parser.parse_args()

    print(f"🚀 STARTING HUMAN RULER (High-Res Mode)...")
    try:
        model = YOLO("yolov8m-pose.pt")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(args.input)
    
    # Check actual video dimensions
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f"   Input Resolution: {int(width)}px width")
    
    # Use 1280 for inference (good balance of speed/accuracy for small objects)
    INFERENCE_SIZE = 1280

    geometry_points = [] 
    frames_processed = 0
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        # Inference at HIGH RES
        try:
            results = model.predict(frame, imgsz=INFERENCE_SIZE, conf=args.conf, verbose=False)[0]
        except Exception:
            continue
        
        valid_in_frame = 0
        
        if results.keypoints is not None:
            # Get keypoints (N, 17, 3)
            kpts = results.keypoints.data.cpu().numpy()
            
            for i, kp in enumerate(kpts):
                # SAFETY: Need enough points
                if kp.shape[0] < 17: continue

                # Relaxed Check: 
                # We need at least ONE foot (15 or 16) and ONE head point (0..4)
                # Conf > 0.15
                feet_conf = np.max(kp[15:17, 2]) 
                head_conf = np.max(kp[0:5, 2])
                
                if feet_conf < 0.15 or head_conf < 0.15: 
                    continue

                min_y = np.min(kp[0:5, 1])   # Head top
                max_y = np.max(kp[15:17, 1]) # Feet bottom
                
                height_px = max_y - min_y
                center_y = max_y 
                
                # Sanity: Height must be reasonable (e.g. > 20px)
                if height_px < 20 or height_px > frame.shape[0]: continue
                if center_y > frame.shape[0]: continue

                geometry_points.append((center_y, height_px))
                valid_in_frame += 1
            
        frames_processed += 1
        if frames_processed % 50 == 0:
            print(f"   Frame {frames_processed}: Found {valid_in_frame} valid skeletons.")
            
        if frames_processed > 200: break 

    cap.release()
    
    # --- ANALYSIS ---
    print(f"📊 Collected {len(geometry_points)} geometry points.")
    
    if len(geometry_points) < 10:
        print("❌ Still not enough data. Try using 'yolov8l-pose.pt' (Large) if available.")
        return

    pts = np.array(geometry_points)
    
    # RANSAC
    from sklearn.linear_model import RANSACRegressor
    X = pts[:, 0].reshape(-1, 1)
    y = pts[:, 1]
    
    try:
        reg = RANSACRegressor(random_state=42, min_samples=0.1).fit(X, y)
        h_top = reg.predict([[200]])[0]
        h_bot = reg.predict([[1080]])[0]
        
        print(f"   Trend: Player @ Top (200px)   ~ {h_top:.1f} px")
        print(f"   Trend: Player @ Bottom (1080px) ~ {h_bot:.1f} px")
        
        if h_bot > h_top:
            print("   ✅ GEOMETRY CONFIRMED: We can mathematically filter coaches.")
        else:
            print("   ⚠️ GEOMETRY WEAK: Perspective not detected.")
            
    except Exception as e:
        print(f"   Analysis failed: {e}")

if __name__ == "__main__":
    main()
