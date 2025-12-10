import json
import os
import cv2
import glob
import sys
import random
from pathlib import Path

sys.path.append(os.getcwd())
from core.vlm_oracle_batch import ask_oracle_batch

MASTER_BANK = "datasets/master_bank"
PROOF_DIR = "runs/viz/oracle_proofs"
TRACKS_DIR = "runs/json"
VIDEO_DIRS = ["tmp_jobs", "runs/videos"]

# CONFIG: High Volume
SAMPLES_PER_VIDEO = 50 
BATCH_SIZE = 5

def get_video_path(job_id):
    for d in VIDEO_DIRS:
        p = f"{d}/{job_id}.mp4"
        if os.path.exists(p): return p
    return None

def process_video(job_id, track_path, video_path):
    print(f"\n📽️ Auditing {job_id}...")
    
    # 1. Load Tracks
    try:
        data = json.loads(Path(track_path).read_text())
        frames = data.get('frames', [])
    except: return []

    # 2. Select Candidates (Smart Sampling)
    # We pick frames where ball speed is high OR players are close (Duels)
    candidates = []
    for i in range(0, len(frames), 10): # Scan every 10th frame
        f = frames[i]
        ball = f.get('ball')
        if not ball: continue
        
        # Calc speed proxy
        speed = 0
        if i > 0:
            prev = frames[i-1].get('ball')
            if prev: speed = abs(ball['x'] - prev['x']) + abs(ball['y'] - prev['y'])
            
        if speed > 0.005: # Moving ball
            candidates.append(i)
            
    # Shuffle and limit
    random.shuffle(candidates)
    selected_frames = candidates[:SAMPLES_PER_VIDEO]
    print(f"   Selected {len(selected_frames)} frames for audit.")

    # 3. Batch Process
    cap = cv2.VideoCapture(video_path)
    new_labels = []
    
    for i in range(0, len(selected_frames), BATCH_SIZE):
        batch_indices = selected_frames[i:i+BATCH_SIZE]
        batch_files = []
        
        # Extract Images
        for f_idx in batch_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            path = f"temp_{job_id}_{f_idx}.jpg"
            cv2.imwrite(path, frame)
            batch_files.append(path)
            
        if not batch_files: continue

        # Ask Oracle
        print(f"   🚀 Sending batch of {len(batch_files)}...")
        labels = ask_oracle_batch(batch_files)
        
        # Save Proofs & Data
        for j, label in enumerate(labels):
            f_idx = batch_indices[j]
            
            # Save Proof Image
            img = cv2.imread(batch_files[j])
            cv2.putText(img, f"AI Label: {label}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            proof_path = f"{PROOF_DIR}/{job_id}_{f_idx}_{label}.jpg"
            cv2.imwrite(proof_path, img)
            
            if label not in ["unknown", "none"]:
                new_labels.append({
                    "frame": f_idx,
                    "label": label,
                    "video": job_id
                })
                
        # Cleanup
        for f in batch_files: os.remove(f)

    cap.release()
    return new_labels

def main():
    os.makedirs(MASTER_BANK, exist_ok=True)
    os.makedirs(PROOF_DIR, exist_ok=True)
    
    track_files = glob.glob(f"{TRACKS_DIR}/fixed_tracks_*.json")
    all_corrections = []
    
    for tf in track_files:
        job_id = Path(tf).stem.replace("fixed_tracks_", "")
        vid = get_video_path(job_id)
        if vid:
            all_corrections.extend(process_video(job_id, tf, vid))
            
    # Save
    if all_corrections:
        out = f"{MASTER_BANK}/oracle_corrections_mass.json"
        Path(out).write_text(json.dumps(all_corrections, indent=2))
        print(f"\n✅ SAVED {len(all_corrections)} NEW LABELS to {out}")
        print(f"🖼️  Proof images saved in {PROOF_DIR}")

if __name__ == "__main__":
    main()
