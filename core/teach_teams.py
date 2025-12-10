import os
import glob
import json
import cv2
import random
import sys
import time
import numpy as np
from pathlib import Path

sys.path.append(os.getcwd())
from core.vlm_oracle import ask_oracle

MASTER_BANK = "datasets/master_bank"
TEAM_DATASET = "datasets/master_bank/team_appearance_db.json"
VIDEO_DIRS = ["tmp_jobs", "runs/videos"]

def get_video_path(job_id):
    for d in VIDEO_DIRS:
        p = f"{d}/{job_id}.mp4"
        if os.path.exists(p): return str(p)
    return None

def ask_oracle_color(image_path):
    # Specialized prompt for Team ID
    import base64
    import requests
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return "unknown"
    
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    prompt = """
    Analyze the player in this crop. Ignore the grass.
    1. What is the PRIMARY color of the shirt? (e.g., "red", "white", "green_stripes").
    2. Is this a Goalkeeper? (yes/no).
    3. Is this a Referee? (yes/no).
    
    Return JSON: {"color": "string", "role": "player/keeper/ref"}
    """
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        "max_tokens": 50,
        "response_format": { "type": "json_object" }
    }
    
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return resp.json()['choices'][0]['message']['content']
    except:
        return "{}"

def main():
    print("👕 STARTING JERSEY SCHOOL (Active Learning for Teams)...")
    os.makedirs(MASTER_BANK, exist_ok=True)
    
    track_files = glob.glob("runs/json/fixed_tracks_*.json")
    new_samples = []
    
    # Load existing DB
    if os.path.exists(TEAM_DATASET):
        try: new_samples = json.loads(Path(TEAM_DATASET).read_text())
        except: pass
        
    print(f"   📚 Current Knowledge Base: {len(new_samples)} player samples.")
    
    for track_file in track_files:
        job_id = Path(track_file).stem.replace("fixed_tracks_", "")
        video_path = get_video_path(job_id)
        if not video_path: continue
        
        print(f"\n   📽️ Scanning {job_id} for kits...")
        
        # Load tracks
        try:
            data = json.loads(Path(track_file).read_text())
            frames = data.get('frames', [])
        except: continue
        
        # Sample 5 random players
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for _ in range(5):
            f_idx = random.randint(0, len(frames)-1)
            frame_data = frames[f_idx]
            players = frame_data.get('players', [])
            if not players: continue
            
            # Pick a player
            p = random.choice(players)
            
            # Get Image
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, img = cap.read()
            if not ret: continue
            
            # Crop
            h, w = img.shape[:2]
            x1 = int(p['x'] * w)
            y1 = int(p['y'] * h)
            # Approx box size (using height heuristics if box not saved)
            # We'll take a 100x100 patch around the center
            crop = img[max(0,y1-50):min(h,y1+50), max(0,x1-30):min(w,x1+30)]
            
            if crop.size == 0: continue
            
            temp_path = "temp_jersey.jpg"
            cv2.imwrite(temp_path, crop)
            
            print(f"      ❓ Asking about Player {p['id']} at frame {f_idx}...")
            resp_str = ask_oracle_color(temp_path)
            time.sleep(1.5)
            
            try:
                resp = json.loads(resp_str)
                color = resp.get('color')
                role = resp.get('role')
                
                if color:
                    print(f"         🎓 Learned: {color} ({role})")
                    new_samples.append({
                        "job_id": job_id,
                        "color_name": color,
                        "role": role,
                        "team_id_assigned": p['team'] # The 'dumb' assignment to verify later
                        # In a real ML loop, we would save the pixel embedding here
                    })
            except: pass
            
        cap.release()
        
    # Save Database
    Path(TEAM_DATASET).write_text(json.dumps(new_samples, indent=2))
    print(f"\n✅ SAVED {len(new_samples)} JERSEY DEFINITIONS.")
    print("   This database will be used to train the Team Classifier.")

if __name__ == "__main__":
    main()
