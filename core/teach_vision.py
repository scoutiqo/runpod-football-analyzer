# core/teach_vision.py
import os
import cv2
import glob
import json
import base64
import requests
import random
from pathlib import Path

DATASET_DIR = Path("datasets/football_vision")
IMG_DIR = DATASET_DIR / "images"
LBL_DIR = DATASET_DIR / "labels"
VIDEO_DIRS = ["tmp_jobs", "runs/videos"]
API_KEY = os.getenv("OPENAI_API_KEY")

def setup():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "data.yaml").write_text(f"path: {DATASET_DIR.absolute()}\ntrain: images\nval: images\nnames:\n  0: player\n  1: ball\n  2: referee\n  3: goalkeeper")

def ask_oracle(img_path):
    with open(img_path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
    prompt = """
    Locate all objects in this football frame.
    Classes: 0=Player, 1=Ball, 2=Referee, 3=Goalkeeper.
    Return JSON: {"objects": [{"class": 0, "box": [x_center, y_center, w, h]}]}
    Coordinates must be normalized (0-1).
    """
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", 
            headers={"Authorization": f"Bearer {API_KEY}"}, 
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], "response_format": {"type": "json_object"}})
        return json.loads(resp.json()['choices'][0]['message']['content']).get('objects', [])
    except: return []

def main():
    print("👁️ STARTING VISION MINER...")
    setup()
    videos = glob.glob("tmp_jobs/*.mp4")
    if not videos: return
    
    count = 0
    for vid in videos:
        cap = cv2.VideoCapture(vid)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Pick 5 frames where we likely missed the ball (random is a good proxy for now)
        for _ in range(5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, total-1))
            ret, frame = cap.read()
            if not ret: continue
            
            path = IMG_DIR / f"train_{count}.jpg"
            cv2.imwrite(str(path), frame)
            
            print(f"   Asking Oracle about image {count}...")
            objs = ask_oracle(path)
            if objs:
                lines = [f"{o['class']} {' '.join(map(str, o['box']))}" for o in objs]
                (LBL_DIR / f"train_{count}.txt").write_text("\n".join(lines))
                print(f"   ✅ Learned {len(objs)} objects.")
                count += 1
            else:
                os.remove(path)
        cap.release()

if __name__ == "__main__":
    main()
