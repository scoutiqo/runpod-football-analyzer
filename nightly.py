import os
import glob
import pandas as pd
import time
from ai_oracle import ask_chatgpt_vision
from data_manager import StorageManager

# --- CONFIG ---
UNCERTAIN_DIR = "tmp_harvest/uncertain"
LABELED_DIR = "tmp_harvest/labeled"
DATASET_FILE = "datasets/master_bank/clean_master_dataset.csv"
TRAINING_SCRIPT = "training/train_master_brain.py"

os.makedirs(UNCERTAIN_DIR, exist_ok=True)
os.makedirs(LABELED_DIR, exist_ok=True)

manager = StorageManager()

def harvest_and_learn():
    print("🌙 Nightly Supervisor Started.")
    
    # 1. Download "Confusing" frames from Cloud
    # (In a real scenario, we'd list them from Backblaze 'uncertain_pool/')
    # For now, we assume worker.py saved them locally to UNCERTAIN_DIR
    
    frames = glob.glob(f"{UNCERTAIN_DIR}/*.jpg")
    if not frames:
        print("💤 No confusing frames found. The model was confident today.")
        return

    print(f"🕵️ Found {len(frames)} confusing frames. Consulting the Oracle...")
    
    new_data = []
    
    for frame_path in frames:
        filename = os.path.basename(frame_path)
        print(f"   Asking about {filename}...")
        
        # PROMPT ENGINEERING
        # We ask ChatGPT to be a data labeler.
        answer = ask_chatgpt_vision(
            frame_path, 
            "You are a football analyst. Look at the player in focus. "
            "Are they performing a 'pass', 'dribble', 'shot', or 'duel'? "
            "Reply with ONLY the single word classification."
        )
        
        if answer:
            print(f"   💡 Oracle says: {answer}")
            
            # Parse filename to get metadata (match_id, frame_idx)
            # Expected format: {match_id}_frame{idx}_{old_label}_conf{score}.jpg
            try:
                parts = filename.split('_')
                match_id = parts[0]
                frame_idx = parts[1].replace("frame", "")
                
                new_data.append({
                    "video_id": match_id,
                    "frame": frame_idx,
                    "label": answer.lower(),
                    "source": "gpt-4o-active-learning"
                })
                
                # Move processed image
                os.rename(frame_path, os.path.join(LABELED_DIR, filename))
                
            except Exception as e:
                print(f"   ⚠️ Filename parse error: {e}")

    # 2. Update Master Dataset
    if new_data:
        df = pd.DataFrame(new_data)
        # Append to CSV (create if doesn't exist)
        header = not os.path.exists(DATASET_FILE)
        df.to_csv(DATASET_FILE, mode='a', header=header, index=False)
        print(f"✅ Added {len(new_data)} new samples to {DATASET_FILE}")
        
        # 3. RETRAIN
        print("🧠 Waking up the Student (Retraining)...")
        exit_code = os.system(f"python {TRAINING_SCRIPT}")
        
        if exit_code == 0:
            print("🚀 Training Successful! Uploading new brain...")
            # Upload the new model to replace the old one
            manager.upload_file("models/event_lstm_master.h5", "models_latest/event_lstm_master.h5")
            print("✅ New Brain Deployed.")
        else:
            print("❌ Training Failed.")
            
    else:
        print("⚠️ Oracle didn't give any clear answers.")

if __name__ == "__main__":
    harvest_and_learn()
