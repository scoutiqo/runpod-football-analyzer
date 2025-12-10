import os
import glob
import shutil
from pathlib import Path

# CONFIG: Look in the correct folder now
FIXED_TRACKS_PATTERN = "runs/json/fixed_tracks_*.json"
MASTER_BANK = "datasets/master_bank"
os.makedirs(MASTER_BANK, exist_ok=True)

def run_command(cmd):
    print(f"   Exec: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"   ❌ Failed.")
        return False
    return True

def main():
    print("🧠 STARTING TOTAL RE-LEARNING SEQUENCE...")
    
    # 1. Find files in the SUBFOLDER
    files = glob.glob(FIXED_TRACKS_PATTERN)
    if not files:
        print(f"❌ No files found matching: {FIXED_TRACKS_PATTERN}")
        return

    print(f"   Found {len(files)} improved datasets to learn from.")
    
    success_count = 0
    
    for track_file in files:
        job_id = Path(track_file).stem.replace("fixed_tracks_", "")
        print(f"\n📘 Processing Job: {job_id}")
        
        # A. MINE NEW LABELS
        # Miner outputs to runs/json/silver_labels.json by default
        cmd_mine = f"python core/events_from_tracks_pipeline.py --input \"{track_file}\""
        if not run_command(cmd_mine): continue
        
        # B. BUILD DATASET
        # We explicitly pass the input tracks and the generated labels
        cmd_build = f"python training/build_event_dataset.py --input_tracks \"{track_file}\" --input_labels runs/json/silver_labels.json"
        if not run_command(cmd_build): continue
        
        # C. HARVEST
        source_csv = "runs/json/event_dataset.csv"
        dest_csv = f"{MASTER_BANK}/{job_id}.csv"
        if os.path.exists(source_csv):
            shutil.copy(source_csv, dest_csv)
            success_count += 1
            print(f"   ✅ Harvested clean knowledge for {job_id}")
        else:
            print("   ⚠️ CSV generation failed.")

    print(f"\n📚 Harvested {success_count}/{len(files)} datasets.")
    
    # 4. MASTER TRAINING
    if success_count > 0:
        print("\n🎓 UPGRADING MASTER BRAIN...")
        run_command("python training/train_master_brain.py")
    else:
        print("\n⚠️ No data harvested, skipping training.")

if __name__ == "__main__":
    main()
