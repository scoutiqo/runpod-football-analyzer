import os
import glob
import pandas as pd
from ai_oracle import ask_oracle
from data_manager import StorageManager

# 1. SETUP
manager = StorageManager()
UNCERTAIN_FRAMES_DIR = "./tmp_harvest/uncertain"
LABELED_DATASET = "datasets/master_bank/clean_master_dataset.csv"

def consult_the_oracle():
    """
    Finds confusing frames, asks ChatGPT, and saves the answer.
    """
    print("🕵️ Hunting for confusing frames...")
    # In a real run, your worker.py would save low-confidence frames here
    frames = glob.glob(f"{UNCERTAIN_FRAMES_DIR}/*.jpg")
    
    if not frames:
        print("✅ No confusing frames found today. Model is confident.")
        return False

    new_labels = []
    for frame in frames:
        print(f"❓ Asking ChatGPT about {os.path.basename(frame)}...")
        
        # The Prompt
        answer = ask_oracle(
            frame, 
            "Look at this football frame. Is the player with the ball performing a 'pass', 'dribble', or 'shot'? Reply with just the word."
        )
        
        if answer:
            print(f"💡 ChatGPT says: {answer}")
            new_labels.append({
                "video_id": "oracle_training",
                "frame": os.path.basename(frame).split('_')[1],
                "label": answer.lower().strip()
            })
            # Move to 'labeled' folder
            # os.rename(frame, frame.replace("uncertain", "labeled"))

    # Save to Master Dataset
    if new_labels:
        df = pd.DataFrame(new_labels)
        df.to_csv(LABELED_DATASET, mode='a', header=False, index=False)
        print(f"✅ Added {len(new_labels)} new expert labels to dataset.")
        return True
    return False

def retrain_brain():
    """
    Triggers the training script to digest the new knowledge.
    """
    print("🧠 Waking up the Student (Training)...")
    # Call your existing training script
    os.system("python training/train_master_brain.py")

if __name__ == "__main__":
    if consult_the_oracle():
        retrain_brain()
        # Upload new model to Backblaze
        manager.upload_file("models/event_lstm_master.h5", "models/event_lstm_master.h5")
        print("🚀 New Brain Deployed to Cloud.")
    else:
        print("💤 No new training needed.")
