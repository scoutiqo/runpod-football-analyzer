import os
import glob
import json
import cv2
import random
import sys
import time

sys.path.append(os.getcwd())
from core.vlm_oracle import ask_oracle

MASTER_BANK = "datasets/master_bank"
VIDEO_DIRS = ["tmp_jobs", "runs/videos"]

def get_video_path(job_id):
    for d in VIDEO_DIRS:
        p = f"{d}/{job_id}.mp4"
        if os.path.exists(p): return p
    return None

def main():
    print("🏫 STARTING 'FULL SYLLABUS' TRAINING...")
    os.makedirs(MASTER_BANK, exist_ok=True)
    
    track_files = glob.glob("runs/json/fixed_tracks_*.json")
    all_new_labels = []
    
    for track_file in track_files:
        job_id = os.path.basename(track_file).replace("fixed_tracks_", "").replace(".json", "")
        video_path = get_video_path(job_id)
        
        if not video_path: continue
        
        print(f"\n📖 Studying Video: {job_id}...")
        
        # Load Tracks
        try:
            data = json.loads(open(track_file).read())
            frames = data.get('frames', [])
            total_frames = len(frames)
        except: continue

        # SAMPLING STRATEGY:
        # To learn "Rare Events" (Tackles, Shots), we pick frames where:
        # 1. Players are very close (Defensive Actions)
        # 2. Ball speed is high (Shots/Crosses/Clearences)
        
        candidates = []
        for i in range(0, total_frames, 50): # Check every 2 seconds
            f = frames[i]
            
            # Check Physics triggers
            ball = f.get('ball')
            players = f.get('players', [])
            
            if not ball: continue
            
            # Find closest player dist
            min_dist = 1.0
            for p in players:
                d = ((p['x']-ball['x'])**2 + (p['y']-ball['y'])**2)**0.5
                if d < min_dist: min_dist = d
            
            # Trigger: Close proximity (Duel/Tackle/Block range)
            if min_dist < 0.02: 
                candidates.append(i)
        
        # Sample 5 candidates per video to label
        if len(candidates) > 5:
            candidates = random.sample(candidates, 5)
            
        cap = cv2.VideoCapture(video_path)
        
        for frame_idx in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            img_path = f"temp_teach_{job_id}.jpg"
            cv2.imwrite(img_path, frame)
            
            print(f"   ❓ Asking Professor about Frame {frame_idx}...")
            label = ask_oracle(img_path)
            time.sleep(1.5) # Pause
            
            print(f"      🎓 Learned: {label}")
            
            if label and label != "unknown":
                all_new_labels.append({
                    "frame": frame_idx,
                    "label": label,
                    "video": job_id
                })
            
            if os.path.exists(img_path): os.remove(img_path)
            
        cap.release()

    # Save Knowledge
    if all_new_labels:
        out_file = f"{MASTER_BANK}/oracle_syllabus.json"
        
        # Merge
        existing = []
        if os.path.exists(out_file):
            existing = json.loads(open(out_file).read())
        
        existing.extend(all_new_labels)
        
        with open(out_file, "w") as f:
            json.dump(existing, f, indent=2)
            
        print(f"\n✅ SAVED {len(all_new_labels)} NEW CONCEPTS to {out_file}")
        print("   The AI now has examples of Tackles, Clearances, etc.")

if __name__ == "__main__":
    main()
