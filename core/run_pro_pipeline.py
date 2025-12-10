#!/usr/bin/env python3
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from ultralytics import YOLO

# --- IMPORT MODULES ---
sys.path.append(os.getcwd())
from core.tracker_players import PlayerTracker
from core.team_assign_v2 import TeamAssigner 
from core.camera_movement_estimator import CameraMovementEstimator
from core.speed_and_distance_estimator import SpeedAndDistanceEstimator
from core.mono_loco import MonoLocoEstimator 
from core.calibrate import DynamicCalibrator 

def interpolate_ball_tracks(raw_frames, calibrator):
    print("   ⚽ Interpolating Ball Tracks & Physics...")
    ball_positions = []
    for f in raw_frames:
        ball = next((t for t in f['tracks'] if t.get('cls') == 32), None)
        if ball:
            bx, by = (ball['x1'] + ball['x2']) / 2, (ball['y1'] + ball['y2']) / 2
            ball_positions.append({'frame': f['frame'], 'bx': bx, 'by': by, 'present': True})
        else:
            ball_positions.append({'frame': f['frame'], 'bx': np.nan, 'by': np.nan, 'present': False})
            
    df = pd.DataFrame(ball_positions)
    df['bx'] = df['bx'].interpolate(method='linear', limit=25, limit_direction='both')
    df['by'] = df['by'].interpolate(method='linear', limit=25, limit_direction='both')
    
    count_recovered = 0
    for i, row in df.iterrows():
        frame_data = raw_frames[i]
        H = frame_data.get('homography')

        if not np.isnan(row['bx']):
            cx, cy = row['bx'], row['by']
            if not row['present']:
                new_track = {"id": -1, "x1": cx-10, "y1": cy-10, "x2": cx+10, "y2": cy+10, "cls": 32, "conf": 0.5}
                frame_data['tracks'].append(new_track)
                count_recovered += 1

            ball_track = next((t for t in frame_data['tracks'] if t.get('cls') == 32), None)
            if ball_track:
                ball_track['x_m'], ball_track['y_m'] = -1.0, -1.0 
                if H is not None:
                    try:
                        pt = np.array([[[cx, cy]]], dtype=np.float32)
                        dst = cv2.perspectiveTransform(pt, H)
                        ball_track['x_m'] = float(dst[0][0][0])
                        ball_track['y_m'] = float(dst[0][0][1])
                    except: pass
    
    print(f"   ✅ Recovered {count_recovered} ball frames.")
    return raw_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--match_id", required=True)
    parser.add_argument("--pitch_mask", default=None)
    parser.add_argument("--save_tracks", default="runs/json/tracks.json")
    args = parser.parse_args()
    
    print(f"🚀 STARTING PRO PIPELINE (SMART FILTER) for {args.match_id}")
    
    model = YOLO("yolov8x.pt") 
    mono_loco = MonoLocoEstimator()
    calibrator = DynamicCalibrator("models/pitch_calibration_v1.pt")
    tracker = PlayerTracker(conf_thresh=0.1, person_ids=(0,)) 
    team_assigner = TeamAssigner()
    speed_estimator = SpeedAndDistanceEstimator()
    
    cap = cv2.VideoCapture(args.video)
    
    raw_frames = []
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # --- NEW: GREEN FILTER (Prevents Tunnel/Crowd Tracking) ---
        # Downscale for speed check
        small_frame = cv2.resize(frame, (640, 360))
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        # Green range for pitch
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = cv2.countNonZero(mask) / mask.size
        
        if green_ratio < 0.35:
            # Not a football pitch (Tunnel, Crowd, Interview)
            raw_frames.append({"frame": frame_idx, "tracks": [], "homography": None})
            frame_idx += 1
            if frame_idx % 100 == 0: print(f"   Skipped {frame_idx} (Non-gameplay)...")
            continue
        # -----------------------------------------------------------
        
        # A. Calibration
        homography = calibrator.calibrate(frame)
        
        # B. 3D Detection
        players_3d = mono_loco.predict(frame)
        
        # C. Tracking
        det_boxes = [p['box'] for p in players_3d] if players_3d else []
        if not det_boxes:
            # Fallback to 2D Detection if 3D fails but pitch is green
            res = model.predict(frame, conf=0.1, classes=[0], verbose=False)[0]
            det_boxes = res.boxes.xyxy.cpu().numpy()

        if len(det_boxes) > 0:
            final_tracks = tracker.update({
                "xyxy": np.array(det_boxes),
                "conf": np.array([0.9]*len(det_boxes)),
                "cls": np.array([0]*len(det_boxes))
            })
        else:
            final_tracks = []
            
        # D. Physics
        for t in final_tracks:
            box_h = t['y2'] - t['y1']
            t['x_m'], t['y_m'] = -1.0, -1.0
            
            # Prefer Homography for stability if available
            if homography is not None:
                cx, cy = (t['x1'] + t['x2']) / 2, t['y2']
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                try:
                    dst = cv2.perspectiveTransform(pt, homography)
                    t['x_m'] = float(dst[0][0][0])
                    t['y_m'] = float(dst[0][0][1])
                except: pass
            
            # Fallback to MonoLoco Height logic if Homography fails
            elif box_h > 10:
                d_m = (1200 * 1.75) / box_h
                img_cx = frame.shape[1] / 2
                box_cx = (t['x1'] + t['x2']) / 2
                x_m = (box_cx - img_cx) * d_m / 1200
                t['x_m'] = float(x_m + 52.5)
                t['y_m'] = float(d_m)

        # E. Ball
        res = model.predict(frame, conf=0.05, imgsz=1280, classes=[32], verbose=False)[0]
        for box in res.boxes:
            b = box.xyxy[0].cpu().numpy()
            final_tracks.append({
                "id": -1, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "cls": 32, "conf": float(box.conf),
                "x_m": -1.0, "y_m": -1.0
            })

        # F. Teams
        people = [t for t in final_tracks if int(t.get('cls',0)) == 0]
        team_assigner.observe(frame, people)
        
        raw_frames.append({"frame": frame_idx, "tracks": final_tracks, "homography": homography})
        frame_idx += 1
        if frame_idx % 50 == 0: print(f"   Processed {frame_idx} frames...")

    cap.release()
    
    raw_frames = interpolate_ball_tracks(raw_frames, calibrator)
    team_assigner.fit_global()
    
    final_export = []
    h, w = (1080, 1920) # Default if read failed, usually overwritten
    
    for i, f_data in enumerate(raw_frames):
        clean_players = []
        clean_ball = None
        for tr in f_data['tracks']:
            cls = int(tr.get('cls', 0))
            xm, ym = tr.get('x_m', -1), tr.get('y_m', -1)
            cx, cy = (tr['x1'] + tr['x2']) / 2, (tr['y1'] + tr['y2']) / 2
            nx, ny = float(cx / w), float(cy / h) # Approx normalization if w/h unknown
            
            if cls == 32:
                clean_ball = {"x": round(nx, 4), "y": round(ny, 4), "x_m": round(xm, 2), "y_m": round(ym, 2)}
            else:
                team = team_assigner.get_team(str(tr['id']))
                clean_players.append({
                    "id": str(tr['id']), "team": team,
                    "x": round(nx, 4), "y": round(ny, 4),
                    "x_m": round(xm, 2), "y_m": round(ym, 2)
                })
        final_export.append({"t": i/fps, "players": clean_players, "ball": clean_ball})

    print("🏃 Calculating Speed...")
    formatted_json = {"fps": fps, "frames": final_export}
    formatted_json = speed_estimator.add_speed_and_distance_to_tracks(formatted_json)
    
    Path(args.save_tracks).write_text(json.dumps(formatted_json))
    print(f"✅ PRO PIPELINE COMPLETE. Saved: {args.save_tracks}")

if __name__ == "__main__":
    main()
