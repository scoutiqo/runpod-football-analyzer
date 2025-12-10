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
DATASET_DIR = Path("datasets/ball_vision")
IMG_DIR = DATASET_DIR / "images"
LBL_DIR = DATASET_DIR / "labels"
VIDEO_DIR = "tmp_jobs" # Look at recent uploads first
API_KEY = os.getenv("OPENAI_API_KEY")

def setup_dirs():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    
    yaml_content = f"""
    path: {os.path.abspath(DATASET_DIR)}
    train: images
    val: images
    names:
      0: ball
    """
    (DATASET_DIR / "data.yaml").write_text(yaml_content)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def ask_gpt_ball(img_path):
    if not API_KEY: return None
    b64 = encode_image(img_path)
    
    prompt = """
    Find the football (soccer ball) in this image.
    Return the Bounding Box in YOLO format: [x_center, y_center, width, height].
    All values 0.0 to 1.0.
    If NO ball is visible, return {"box": null}.
    Response JSON: {"box": [0.5, 0.5, 0.02, 0.03]}
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
        return json.loads(resp.json()['choices'][0]['message']['content']).get("box")
    except: return None

def main():
    print("⚽ STARTING BALL VISION MINER...")
    setup_dirs()
    
    videos = glob.glob(f"{VIDEO_DIR}/*.mp4")
    if not videos:
        print(f"❌ No videos found in {VIDEO_DIR}. Upload a video first.")
        return
        
    count = 0
    for vid in videos:
        if count >= 60: break # We need ~50-60 samples for a good start
        
        print(f"   Scanning {os.path.basename(vid)}...")
        cap = cv2.VideoCapture(vid)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Pick 10 random frames per video
        frames = random.sample(range(0, total), 10)
        
        for f_idx in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            img_name = f"{Path(vid).stem}_{f_idx}.jpg"
            img_path = IMG_DIR / img_name
            cv2.imwrite(str(img_path), frame)
            
            print(f"      Asking about frame {f_idx}...")
            box = ask_gpt_ball(str(img_path))
            
            if box:
                print(f"      ✅ Found Ball: {box}")
                lbl_path = LBL_DIR / f"{Path(vid).stem}_{f_idx}.txt"
                lbl_path.write_text(f"0 {box[0]} {box[1]} {box[2]} {box[3]}")
                count += 1
            else:
                print("      x No ball.")
                if os.path.exists(img_path): os.remove(img_path)
                
        cap.release()

    print(f"\n✅ Collected {count} ball samples.")
    print("   👉 Run 'python core/train_ball_model.py' to train the Specialist.")

if __name__ == "__main__":
    main()
