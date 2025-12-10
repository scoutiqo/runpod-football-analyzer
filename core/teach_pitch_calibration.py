import os
import json
import cv2
import glob
import base64
import requests
import time
import random
import numpy as np
from pathlib import Path

# CONFIG
DATASET_DIR = Path("datasets/pitch_calibration")
IMG_DIR = DATASET_DIR / "images"
LBL_DIR = DATASET_DIR / "labels_json"
VIDEO_DIRS = ["tmp_jobs", "runs/videos"]
TRACKS_DIR = "runs/json"

API_KEY = os.getenv("OPENAI_API_KEY")

KEYPOINTS = [
    "TL_Corner", "TR_Corner", "BR_Corner", "BL_Corner",
    "Center_Circle_Top", "Center_Circle_Bottom", "Center_Spot",
    "Penalty_Spot_Left", "Penalty_Spot_Right",
    "Box_TL_Left", "Box_BL_Left", "Box_TR_Right", "Box_BR_Right"
]

def setup_dirs():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_video_path(job_id):
    for d in VIDEO_DIRS:
        p = f"{d}/{job_id}.mp4"
        if os.path.exists(p): return p
    return None

def ask_surveyor(img_path):
    if not API_KEY: return None
    b64 = encode_image(img_path)
    
    prompt = f"""
    Analyze this football pitch image. Identify visible keypoints: {KEYPOINTS}.
    Return normalized coordinates [x, y] (0.0-1.0).
    Response JSON: {{"points": {{"Center_Spot": [0.5, 0.5]}}}}
    """
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}],
        "response_format": { "type": "json_object" }
    }
    
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return json.loads(resp.json()['choices'][0]['message']['content']).get("points", {})
    except: return None

def main():
    print("📐 STARTING CONTINUOUS PITCH SURVEY...")
    setup_dirs()
    
    track_files = glob.glob(f"{TRACKS_DIR}/fixed_tracks_*.json")
    if not track_files: return

    # Randomly pick ONE video to audit per cycle (to keep loop fast)
    tf = random.choice(track_files)
    job_id = Path(tf).stem.replace("fixed_tracks_", "")
    vid_path = get_video_path(job_id)
    
    if not vid_path: 
        print("   ⚠️ Video not found.")
        return
        
    print(f"   Surveying {job_id}...")
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Pick 5 random NEW frames
    candidates = random.sample(range(0, total_frames), 5)
    new_samples = 0
    
    for f_idx in candidates:
        # Check if we already did this frame (skip duplicates)
        json_path = LBL_DIR / f"{job_id}_{f_idx}.json"
        if json_path.exists(): continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        img_name = f"{job_id}_{f_idx}.jpg"
        img_path = IMG_DIR / img_name
        cv2.imwrite(str(img_path), frame)
        
        points = ask_surveyor(str(img_path))
        time.sleep(1.0)
        
        if points:
            print(f"      ✅ Learned Frame {f_idx} ({len(points)} points)")
            json_path.write_text(json.dumps(points, indent=2))
            new_samples += 1
        else:
            if os.path.exists(img_path): os.remove(img_path)

    cap.release()
    print(f"   📝 Added {new_samples} new calibration samples.")

if __name__ == "__main__":
    main()
