import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm
import os

TRACKS_FILE = "runs/json/tracks.json"
VIDEO_DIR = "tmp_jobs"
MODEL_PATH = "models/jersey_ocr_v1.pt"

def main():
    print("🎽 APPLYING JERSEY OCR (Identity Fusion)...")
    
    if not Path(TRACKS_FILE).exists() or not Path(MODEL_PATH).exists():
        print("❌ Missing tracks or model.")
        return

    # 1. Load Data
    data = json.loads(Path(TRACKS_FILE).read_text())
    frames = data['frames']
    
    # Find video
    videos = list(Path(VIDEO_DIR).glob("*.mp4"))
    if not videos: return
    vid_path = max(videos, key=lambda p: p.stat().st_mtime)
    
    # Load Model (Ensure you have it, else fallback)
    try:
        model = YOLO(MODEL_PATH)
    except:
        print("   ⚠️ Custom OCR model failed. Falling back to yolov8n-cls.")
        model = YOLO("yolov8n-cls.pt")

    cap = cv2.VideoCapture(str(vid_path))
    track_numbers = defaultdict(list)
    
    # 2. Scan Video
    step = 5
    # Use enumerate to get frame index safely
    for i in tqdm(range(0, len(frames), step)):
        f_data = frames[i]
        
        # FIX: Use implicit index 'i' since 'frame' key might be missing
        frame_idx = f_data.get('frame', i)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        if not ret: break
        
        h_img, w_img = img.shape[:2]
        
        for p in f_data['players']:
            pid = str(p['id'])
            cx, cy = p['x'], p['y']
            
            # Dynamic crop
            scale = 0.05 + (cy * 0.08) 
            w_box = scale * w_img
            h_box = w_box * 2.0
            
            x1 = int((cx * w_img) - w_box/2)
            y1 = int((cy * h_img) - h_box/2)
            x2 = int((cx * w_img) + w_box/2)
            y2 = int((cy * h_img) + h_box/2)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            if (y2-y1) < 30 or (x2-x1) < 10: continue
            
            crop = img[y1:y2, x1:x2]
            back_crop = crop[0:int((y2-y1)*0.5), :]
            
            if back_crop.size == 0: continue
            
            # Inference
            results = model(back_crop, verbose=False)
            
            if results and results[0].probs:
                top1_idx = results[0].probs.top1
                conf = results[0].probs.top1conf.item()
                if conf > 0.85:
                    # If using generic classification, this might just be a class index
                    # If using custom trained, it's the number string
                    number = results[0].names[top1_idx]
                    track_numbers[pid].append(number)

    cap.release()
    
    # 3. Consolidate
    print(f"   📊 Found numbers for {len(track_numbers)} tracks.")
    final_map = {}
    for pid, nums in track_numbers.items():
        from collections import Counter
        counts = Counter(nums)
        best_num, count = counts.most_common(1)[0]
        if count >= 3 or (count/len(nums) > 0.5):
            final_map[pid] = best_num
            
    # 4. Inject
    for f in frames:
        for p in f['players']:
            pid = str(p['id'])
            if pid in final_map:
                p['jersey_number'] = final_map[pid]
                
    Path(TRACKS_FILE).write_text(json.dumps(data))
    print("   💾 Saved Jersey Numbers to tracks.json")

if __name__ == "__main__":
    main()
