#!/usr/bin/env python3
"""
run_tracker_cli.py (V7 - Gameplay Aware)

1. Checks if frame is "Tactical View" (Green Dominant).
2. Skips Ads/Replays/Close-ups.
3. Tracks/Assigns teams only on valid frames.
"""

import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tracker_players import PlayerTracker
from team_assign_manual import ManualTeamAssigner
from team_assign_v2 import TeamAssigner as AutoAssigner

def is_gameplay_view(frame):
    """
    Returns True if the frame looks like a wide-angle football pitch.
    Heuristic: Is > 30% of the image green?
    """
    # Downscale for speed
    small = cv2.resize(frame, (128, 72))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    
    # Green range (Broad)
    lower_green = np.array([25, 25, 25])
    upper_green = np.array([95, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.count_nonzero(mask) / mask.size
    
    # If < 25% is green, it's likely a close-up, crowd, or ad.
    return green_ratio > 0.25

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--save", default="runs/json/tracks_wrapped_short_pb_ball.json")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--color_a", default=None)
    ap.add_argument("--color_b", default=None)
    
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists(): sys.exit(1)
    
    ensure_dirs()

    use_manual = (args.color_a and args.color_b)
    if use_manual:
        print(f"🚀 STARTING TRACKER (Manual: A={args.color_a}, B={args.color_b})")
        assigner = ManualTeamAssigner(args.color_a, args.color_b)
    else:
        print(f"🚀 STARTING TRACKER (Auto-Clustering)")
        assigner = AutoAssigner()

    cap = cv2.VideoCapture(str(src))
    model = YOLO(args.model)
    tracker = PlayerTracker(conf_thresh=args.conf, person_ids=(0, 32))

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracks_frames = [] 
    t_idx = 0
    
    skipped_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        
        # --- GAMEPLAY FILTER ---
        # If it's an Ad or Replay, skip tracking to avoid polluting data
        if not is_gameplay_view(frame):
            tracks_frames.append({"t": t_idx/fps, "ball": None, "players": []})
            skipped_frames += 1
            t_idx += 1
            continue
        # -----------------------

        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        
        det = {
            "xyxy": res.boxes.xyxy.cpu().numpy(),
            "conf": res.boxes.conf.cpu().numpy(),
            "cls": res.boxes.cls.cpu().numpy().astype(int)
        }
        tr = tracker.update(det) 
        
        if not use_manual:
            people = [t for t in tr if int(t.get('cls', 0)) == 0]
            assigner.observe(frame, people)

        players = []
        ball = None
        
        for r in tr:
            c = int(r.get("cls", 0))
            bbox = [r["x1"], r["y1"], r["x2"], r["y2"]]
            
            if c == 32: 
                 if ball is None or r.get("conf",0) > ball.get("conf",0):
                    ball = {"x": (r["x1"]+r["x2"])/2, "y": (r["y1"]+r["y2"])/2}
            else:
                team = assigner.get_team(frame, bbox) if use_manual else "unknown"
                players.append({
                    "id": int(r["id"]),
                    "x": (r["x1"]+r["x2"])/2, 
                    "y": (r["y1"]+r["y2"])/2,
                    "team": team
                })
        
        tracks_frames.append({"t": t_idx/fps, "ball": ball, "players": players})
        t_idx += 1
        if t_idx % 100 == 0: print(f"   Processed {t_idx} frames... (Skipped {skipped_frames} ads/replays)")

    cap.release()
    
    if not use_manual:
        print("🤖 Finalizing Auto-Clustering...")
        assigner.fit_global()
        for f in tracks_frames:
            for p in f['players']:
                p['team'] = assigner.get_team(str(p['id']))
                if p['team'] == 'unknown': p['team'] = 'A'

    with open(args.save, "w") as f:
        json.dump({"fps": fps, "frames": tracks_frames}, f)
    
    print(f"✅ TRACKING COMPLETE. Skipped {skipped_frames} non-gameplay frames.")

def ensure_dirs():
    (Path("runs/json")).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    main()
