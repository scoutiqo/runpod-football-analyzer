import json
import os
import cv2
import time
from pathlib import Path
import sys

sys.path.append(os.getcwd())
from core.vlm_oracle_batch import ask_oracle_batch

# CONFIG
PREDICTIONS_PATH = "runs/json/predicted_events_learned.json"
VIDEO_PATH = "viewer/test_short.mp4"
MASTER_BANK = "datasets/master_bank"
CONFIDENCE_THRESHOLD = 0.60
BATCH_SIZE = 5 # Process 5 images at once

def main():
    print(f"🎓 Starting BATCH Active Learning (Threshold: {CONFIDENCE_THRESHOLD})...")
    
    if not os.path.exists(PREDICTIONS_PATH): return

    preds = json.loads(Path(PREDICTIONS_PATH).read_text())
    
    # 1. Filter Low Confidence
    candidates = [p for p in preds if p['prob'] < CONFIDENCE_THRESHOLD]
    print(f"   Found {len(candidates)} candidates.")
    
    if not candidates: return

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    current_batch_files = []
    current_batch_indices = []
    new_labels = []
    
    # Process in Batches
    for i, evt in enumerate(candidates):
        frame_idx = evt['frame']
        
        # Extract
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        img_path = f"temp_batch_{i}.jpg"
        cv2.imwrite(img_path, frame)
        
        current_batch_files.append(img_path)
        current_batch_indices.append(frame_idx)
        
        # If Batch Full, Execute
        if len(current_batch_files) >= BATCH_SIZE:
            print(f"   🚀 Sending Batch of {BATCH_SIZE} to Teacher...")
            
            try:
                labels = ask_oracle_batch(current_batch_files)
                
                # Map back results
                for idx, lbl in enumerate(labels):
                    print(f"      Frame {current_batch_indices[idx]} -> {lbl}")
                    new_labels.append({
                        "frame": current_batch_indices[idx],
                        "label": lbl,
                        "source": "vlm_oracle"
                    })
            except Exception as e:
                print(f"      ❌ Batch failed after retries: {e}")
            
            # Cleanup
            for f in current_batch_files: os.remove(f)
            current_batch_files = []
            current_batch_indices = []
            
            # Polite delay
            time.sleep(1) 

    cap.release()
    
    # Save Logic
    if new_labels:
        os.makedirs(MASTER_BANK, exist_ok=True)
        output_path = f"{MASTER_BANK}/oracle_corrections_batch.json"
        
        existing = []
        if os.path.exists(output_path):
            existing = json.loads(Path(output_path).read_text())
            
        existing.extend(new_labels)
        Path(output_path).write_text(json.dumps(existing, indent=2))
        print(f"✅ Saved {len(new_labels)} new labels.")

if __name__ == "__main__":
    main()
