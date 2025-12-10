import glob
import os
import subprocess

def main():
    print("🚀 Applying Nightly Improvements to the Brain...")
    
    # 1. Find Improved Tracks
    files = glob.glob("runs/json/improved_tracks_*.json")
    if not files:
        print("   No improvements found. Run the Researcher first.")
        return

    count = 0
    for f in files:
        # We treat these 'improved' tracks as the new Ground Truth
        # We run the Miner on them to extract the NEW events (like Shots)
        print(f"   🧠 Re-learning from: {f}")
        
        # 1. Mine (This will now see the interpolated ball and new shot suggestions)
        subprocess.run(f"python core/events_from_tracks_pipeline.py --input {f}", shell=True)
        
        # 2. Build Features
        subprocess.run(f"python training/build_event_dataset.py --input_tracks {f} --input_labels runs/json/silver_labels.json", shell=True)
        
        # 3. Harvest
        job_id = f.split("_")[-1].replace(".json", "")
        subprocess.run(f"cp runs/json/event_dataset.csv datasets/master_bank/{job_id}.csv", shell=True)
        count += 1

    print(f"\n✅ Applied improvements to {count} datasets.")
    print("   You can now run 'python training/train_master_brain.py' to upgrade the LSTM.")

if __name__ == "__main__":
    main()
