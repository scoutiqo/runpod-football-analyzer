import json
import cv2
import os
import random
import base64
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm

# CONFIG
TRACKS_DIR = Path("runs/json")
DATASET_DIR = Path("datasets/jersey_numbers")
IMG_DIR = DATASET_DIR / "images"
LBL_DIR = DATASET_DIR / "labels"
VIDEO_DIR = Path("tmp_jobs")
API_KEY = os.getenv("OPENAI_API_KEY")

def setup():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    if not (DATASET_DIR / "values.csv").exists():
        (DATASET_DIR / "values.csv").write_text("image,number\n")

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def ask_oracle(crop):
    if not API_KEY: return None
    
    # 1. Filter: Reject if too small/blurry
    h, w = crop.shape[:2]
    if h < 40: return None 

    # 2. Upscale for AI readability
    crop = cv2.resize(crop, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    b64 = encode_image(crop)
    
    prompt = """
    Read the Jersey Number (0-99) visible on this player's back or shorts.
    Strict Rules:
    - If blurry or occluded, return null.
    - If side view, return null.
    - Return JSON: {"number": "7"}
    """
    
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", 
            headers={"Authorization": f"Bearer {API_KEY}"}, 
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], "response_format": {"type": "json_object"}}, timeout=5)
        return json.loads(resp.json()['choices'][0]['message']['content']).get("number")
    except: return None

def main():
    print("🎽 STARTING JERSEY MINER (FROM TRACKS)...")
    setup()
    
    # Find the big track files
    files = list(TRACKS_DIR.glob("fixed_tracks_*.json"))
    if not files: return

    for tf in files:
        job_id = tf.stem.replace("fixed_tracks_", "")
        
        # Find video
        vid_path = VIDEO_DIR / f"{job_id}.mp4"
        if not vid_path.exists():
            # Try searching widely
            candidates = list(VIDEO_DIR.glob(f"*{job_id}*.mp4"))
            if candidates: vid_path = candidates[0]
            else: continue

        print(f"   Mining {vid_path.name}...")
        try:
            data = json.loads(tf.read_text())
            frames = data['frames']
        except: continue
        
        cap = cv2.VideoCapture(str(vid_path))
        
        # Pick 100 random frames to sample
        indices = sorted(random.sample(range(len(frames)), min(100, len(frames))))
        
        for f_idx in tqdm(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            h_img, w_img = frame.shape[:2]
            frame_data = frames[f_idx]
            
            for p in frame_data['players']:
                # Get normalized coords
                cx, cy = p['x'], p['y']
                
                # Estimate Box (Heuristic based on typical player aspect ratio)
                # Far away players are smaller
                p_h_norm = 0.15 if cy > 0.5 else 0.08 
                p_w_norm = p_h_norm * 0.4
                
                x1 = int((cx - p_w_norm/2) * w_img)
                y1 = int((cy - p_h_norm/2) * h_img)
                x2 = int((cx + p_w_norm/2) * w_img)
                y2 = int((cy + p_h_norm/2) * h_img)
                
                # Clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                if (y2 - y1) < 30: continue # Too small
                
                crop = frame[y1:y2, x1:x2]
                
                # Crop Upper Back (Top 60%)
                back_crop = crop[0:int(crop.shape[0]*0.6), :]
                
                num = ask_oracle(back_crop)
                if num:
                    fname = f"{job_id}_{f_idx}_{p['id']}"
                    cv2.imwrite(str(IMG_DIR / f"{fname}.jpg"), back_crop)
                    # Save dummy YOLO label for now
                    (LBL_DIR / f"{fname}.txt").write_text(f"0 0.5 0.5 1.0 1.0") 
                    (DATASET_DIR / "values.csv").open("a").write(f"{fname}.jpg,{num}\n")
        
        cap.release()

if __name__ == "__main__":
    main()
