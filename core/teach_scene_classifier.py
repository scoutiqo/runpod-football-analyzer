import os
import cv2
import json
import base64
import requests
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# CONFIG
DATASET_DIR = Path("datasets/scene_classification")
VIDEO_DIR = Path("tmp_jobs")
API_KEY = os.getenv("OPENAI_API_KEY")

CLASSES = ["wide", "close", "junk"]

def setup():
    for split in ["train", "val"]:
        for c in CLASSES:
            (DATASET_DIR / split / c).mkdir(parents=True, exist_ok=True)

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def ask_director(img):
    if not API_KEY: return None
    img = cv2.resize(img, (512, 512))
    b64 = encode_image(img)
    
    prompt = """
    Classify this football broadcast frame. Be extremely critical.
    
    1. "wide": ONLY if it is a clear, far-out tactical view of the pitch where game flow is visible.
    2. "close": Zoomed in on 1-3 players, referee, or coach.
    3. "junk": Crowd shots, replays, logos, VAR screens, blurry transitions, or empty grass.
    
    Return JSON: {"class": "wide"}
    """
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", 
            headers={"Authorization": f"Bearer {API_KEY}"}, 
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], "response_format": {"type": "json_object"}}, timeout=10)
        
        content = resp.json()['choices'][0]['message']['content']
        return json.loads(content).get("class", "").lower()
    except: return None

def main():
    print("🎬 STARTING SCENE DIRECTOR TRAINING (BRUTE FORCE)...")
    setup()
    
    videos = list(VIDEO_DIR.glob("*.mp4"))
    if not videos: return

    counts = {c: 0 for c in CLASSES}
    
    for vid in videos:
        # If we have enough data, stop
        if all(c > 60 for c in counts.values()): break
        
        print(f"   Scanning {vid.name}...")
        cap = cv2.VideoCapture(str(vid))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Scan 30 random frames per video without ANY pre-filtering
        for _ in range(30):
            f_idx = random.randint(0, total-1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            # Ask Oracle directly
            label = ask_director(frame)
            
            if label in CLASSES:
                # Balance the dataset: Don't save more than 100 "wide" shots
                if label == "wide" and counts['wide'] > 100: continue
                
                split = "train" if random.random() < 0.8 else "val"
                fname = f"{vid.stem}_{f_idx}.jpg"
                save_path = DATASET_DIR / split / label / fname
                cv2.imwrite(str(save_path), frame)
                counts[label] += 1
                print(f"      ✅ Classified as {label.upper()}")
            else:
                print(f"      ⚠️ Unknown class: {label}")
                
        cap.release()

    print(f"\n✅ Data Mining Complete: {counts}")
    print("   👉 Run 'python core/train_scene_model.py' next.")

if __name__ == "__main__":
    main()
