import json
import os
import cv2
import glob
import sys
import math
import random
import time
from pathlib import Path

# Add path to find vlm_oracle
sys.path.append(os.getcwd())
from core.vlm_oracle import ask_oracle

# CONFIG
MASTER_BANK = "datasets/master_bank"
TRACKS_DIR = "runs/json"
VIDEO_DIRS = ["tmp_jobs", "runs/videos"]

def get_video_path(job_id):
    for d in VIDEO_DIRS:
        p = Path(d) / f"{job_id}.mp4"
        if p.exists(): return str(p)
    return None

def find_interesting_frames(track_path, limit=15):
    """
    Physics-based hunter for rare events.
    """
    try:
        data = json.loads(Path(track_path).read_text())
        frames = data.get('frames', [])
    except: return []

    candidates = []
    
    for i in range(1, len(frames)-1):
        f = frames[i]
        ball = f.get('ball')
        players = f.get('players', [])
        
        if not ball: continue
        
        # 1. COLLISION DETECTION (Tackles, Fouls, Blocks)
        # Find distance to nearest 2 players
        dists = []
        for p in players:
            d = math.sqrt((p['x']-ball['x'])**2 + (p['y']-ball['y'])**2)
            dists.append(d)
        dists.sort()
        
        # If 2 players are extremely close to ball (< 2% screen width) -> DUEL/TACKLE
        if len(dists) >= 2 and dists[1] < 0.02:
            candidates.append((i, "collision"))
            continue
            
        # 2. HIGH SPEED NEAR GOAL (Shots, Saves, Goal Kicks)
        # Normalized X < 0.1 or > 0.9 is goal area
        bx, by = ball['x'], ball['y']
        prev = frames[i-1].get('ball')
        
        if prev:
            speed = math.sqrt((bx-prev['x'])**2 + (by-prev['y'])**2)
            if speed > 0.02 and (bx < 0.1 or bx > 0.9):
                candidates.append((i, "goal_action"))
                continue
    
    # Randomly sample from the findings to get diversity
    if not candidates: return []
    
    random.shuffle(candidates)
    return candidates[:limit]

def process_video(job_id, track_path, video_path):
    print(f"\n📽️  Mining Knowledge from: {job_id}...")
    
    # 1. Find Physics Events
    targets = find_interesting_frames(track_path, limit=10) # 10 events per video
    print(f"    Found {len(targets)} physics-of-interest frames (Collisions/Shots).")
    
    cap = cv2.VideoCapture(video_path)
    new_labels = []
    
    for frame_idx, reason in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        # Save temp
        temp_img = f"temp_force_{job_id}.jpg"
        cv2.imwrite(temp_img, frame)
        
        print(f"    ❓ Analyzing Frame {frame_idx} ({reason})...")
        
        try:
            # Ask the Expert (GPT-4o)
            label = ask_oracle(temp_img)
            time.sleep(1) # Safety pause
            
            if label and label != "unknown" and label != "none":
                print(f"       🎓 Learned: {label.upper()}")
                new_labels.append({
                    "frame": frame_idx,
                    "label": label,
                    "video": job_id,
                    "source": "force_learn"
                })
        except Exception as e:
            print(f"       ⚠️ Error: {e}")
            
        if os.path.exists(temp_img): os.remove(temp_img)

    cap.release()
    return new_labels

def main():
    print("🏫 STARTING FORCE-LEARNING PROTOCOL...")
    os.makedirs(MASTER_BANK, exist_ok=True)
    
    track_files = glob.glob(f"{TRACKS_DIR}/fixed_tracks_*.json")
    all_knowledge = []
    
    for tf in track_files:
        job_id = Path(tf).stem.replace("fixed_tracks_", "")
        vid = get_video_path(job_id)
        if vid:
            knowledge = process_video(job_id, tf, vid)
            all_knowledge.extend(knowledge)
            
    # Save Syllabus
    if all_knowledge:
        out_path = f"{MASTER_BANK}/oracle_syllabus_pro.json"
        
        # Load existing
        existing = []
        if os.path.exists(out_path):
            try: existing = json.loads(Path(out_path).read_text())
            except: pass
            
        existing.extend(all_knowledge)
        Path(out_path).write_text(json.dumps(existing, indent=2))
        
        print(f"\n✅ SAVED {len(all_knowledge)} NEW PROFESSIONAL CONCEPTS.")
        print(f"   File: {out_path}")
        print("   👉 Run 'python training/train_master_brain.py' to finalize.")
    else:
        print("\n❌ No new knowledge found. Check video paths.")

if __name__ == "__main__":
    main()
