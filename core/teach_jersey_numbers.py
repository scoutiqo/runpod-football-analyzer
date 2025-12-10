import os
import cv2
import json
import base64
import requests
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# CONFIG
DATASET_DIR = Path("datasets/jersey_numbers")
IMG_DIR = DATASET_DIR / "images"
LBL_DIR = DATASET_DIR / "labels"
DEBUG_DIR = DATASET_DIR / "debug_rejects" # New folder for failures
VIDEO_DIR = "tmp_jobs"
API_KEY = os.getenv("OPENAI_API_KEY")

def setup():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "data.yaml").write_text(f"path: {DATASET_DIR.absolute()}\ntrain: images\nval: images\nnames:\n  0: number")

def encode_image(img_array):
    # Encode numpy array directly to base64
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

def ask_oracle_number(crop):
    if not API_KEY: return None
    
    # Super-Resolution (Simple Upscale) to help OCR
    # Resize 2x to help the model see structure
    h, w = crop.shape[:2]
    upscaled = cv2.resize(crop, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    
    b64 = encode_image(upscaled)
    
    prompt = """
    Look at this crop of a football player's back/shorts.
    Can you clearly read a Jersey Number?
    
    Strict Rules:
    - If it is blurry, return null.
    - If it is side-view, return null.
    - Only return if you are 90% sure.
    
    Return JSON: {"number": "10"} or {"number": null}
    """
    
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", 
            headers={"Authorization": f"Bearer {API_KEY}"}, 
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], "response_format": {"type": "json_object"}},
            timeout=10)
        return json.loads(resp.json()['choices'][0]['message']['content']).get("number")
    except: return None

def main():
    print("🎽 STARTING JERSEY FORENSICS MINER...")
    setup()
    
    videos = list(Path(VIDEO_DIR).glob("*.mp4"))
    if not videos: 
        print("❌ No videos found.")
        return

    model = YOLO("yolov8x.pt") # Use largest model for best detection
    
    count = 0
    reject_count = 0
    
    for vid in videos:
        if count > 50: break
        print(f"   Scanning {vid.name}...")
        cap = cv2.VideoCapture(str(vid))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Scan 15 frames
        for _ in range(15):
            f_idx = random.randint(0, total-1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            # Detect Persons
            res = model.predict(frame, conf=0.1, classes=[0], verbose=False)[0]
            
            for box in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                h = y2 - y1
                w = x2 - x1
                
                # FILTER: Ignore small players (Too far away to read)
                if h < 80: continue 
                
                # Crop Upper Back
                crop = frame[y1:y1+int(h*0.6), x1:x2]
                
                # Ask Oracle
                print(f"      🔎 Analyzing crop (Height: {h}px)...")
                number = ask_oracle_number(crop)
                
                if number:
                    print(f"         ✅ Found #{number}")
                    fname = f"{vid.stem}_{f_idx}_{count}"
                    cv2.imwrite(str(IMG_DIR / f"{fname}.jpg"), crop)
                    (LBL_DIR / f"{fname}.txt").write_text(f"0 0.5 0.5 1.0 1.0") # Placeholder box
                    (DATASET_DIR / "values.csv").open("a").write(f"{fname}.jpg,{number}\n")
                    count += 1
                else:
                    # Save REJECT to inspect why it failed
                    if reject_count < 20:
                        fname = f"REJECT_{vid.stem}_{f_idx}_{reject_count}.jpg"
                        cv2.imwrite(str(DEBUG_DIR / fname), crop)
                        reject_count += 1
                    print("         x Rejected (Blurry/Angle)")
                    
    cap.release()
    print(f"\n✅ Collected {count} Valid Samples.")
    print(f"⚠️ Saved {reject_count} Rejected Samples to {DEBUG_DIR} for inspection.")

if __name__ == "__main__":
    main()
