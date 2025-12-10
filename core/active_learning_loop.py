import time
import json
import os
import cv2
import glob
import sys
import subprocess
import random
from pathlib import Path

# Add project root to path to find modules
sys.path.append(os.getcwd())
from core.vlm_oracle import ask_oracle

# CONFIG
MASTER_BANK = "datasets/master_bank"
CONFIDENCE_THRESHOLD = 0.80
TRACKS_DIR = "runs/json"
VIDEO_DIRS = ["tmp_jobs", "runs/videos"]

def get_video_path(job_id):
    for d in VIDEO_DIRS:
        path = Path(d) / f"{job_id}.mp4"
        if path.exists(): return str(path)
    return None

def process_video(job_id, track_path, video_path):
    print(f"\n📽️ Deep Mining Job: {job_id}...")
    
    if not os.path.exists(video_path):
        return []

    # 1. Inference (Generate Predictions)
    python_exe = sys.executable
    temp_preds = f"temp_preds_{job_id}.json"
    cmd = f"{python_exe} core/predict_events_lstm.py --tracks {track_path} --output {temp_preds}"
    
    try:
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except: pass
    
    candidates = []
    if os.path.exists(temp_preds):
        try:
            preds = json.loads(Path(temp_preds).read_text())
            # Priority 1: Confusing Frames
            candidates = [p['frame'] for p in preds if p['prob'] < CONFIDENCE_THRESHOLD]
            print(f"   Found {len(candidates)} low-confidence frames.")
        except: pass
    
    # Priority 2: Random Sampling (The "Deep Mine")
    # If we have < 50 candidates, fill the rest with random frames from the video
    # This ensures we eventually check EVERY part of the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    target_count = 20 # Check 20 frames per pass
    while len(candidates) < target_count:
        rnd = random.randint(0, total_frames - 1)
        if rnd not in candidates:
            candidates.append(rnd)
            
    # Shuffle and limit
    random.shuffle(candidates)
    selected = candidates[:target_count]
    
    print(f"   🚀 Sending {len(selected)} frames to The Professor...")

    new_labels = []
    
    for frame_idx in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        temp_img = f"temp_audit_{job_id}.jpg"
        cv2.imwrite(temp_img, frame)
        
        try:
            label = ask_oracle(temp_img)
            time.sleep(1.5) # Be polite to API
            
            # Only save USEFUL labels (ignore empty space)
            if label not in ["unknown", "none", "null"]:
                print(f"      🎓 Learned: {label.upper()} (Frame {frame_idx})")
                new_labels.append({
                    "frame": frame_idx,
                    "label": label,
                    "video": job_id,
                    "source": "deep_mine"
                })
            else:
                print(f"      . (Nothing at {frame_idx})")
                
        except Exception as e:
            print(f"      ⚠️ Error: {e}")
            
        if os.path.exists(temp_img): os.remove(temp_img)
            
    cap.release()
    if os.path.exists(temp_preds): os.remove(temp_preds)
    return new_labels

def main():
    os.makedirs(MASTER_BANK, exist_ok=True)
    
    # Find tracks
    track_files = glob.glob(f"{TRACKS_DIR}/fixed_tracks_*.json")
    if not track_files: return

    all_corrections = []
    
    # Pick ONE random video to process per run (to distribute load)
    # Or iterate all if you have budget
    track_file = random.choice(track_files)
    job_id = Path(track_file).stem.replace("fixed_tracks_", "")
    vid = get_video_path(job_id)
    
    if vid:
        corrections = process_video(job_id, track_file, vid)
        all_corrections.extend(corrections)

    if all_corrections:
        # Append to the growing syllabus
        output_path = f"{MASTER_BANK}/oracle_syllabus_deep.json"
        existing = []
        if os.path.exists(output_path):
            try: existing = json.loads(Path(output_path).read_text())
            except: pass
        
        existing.extend(all_corrections)
        
        Path(output_path).write_text(json.dumps(existing, indent=2))
        print(f"\n✅ SAVED {len(all_corrections)} NEW CONCEPTS.")
        
if __name__ == "__main__":
    main()
