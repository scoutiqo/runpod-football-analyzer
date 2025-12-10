import pandas as pd
import glob
import os
from pathlib import Path

MASTER_DIR = "datasets/master_bank"
OUTPUT_FILE = "datasets/master_bank/clean_master_dataset.csv"

def main():
    print("🧹 CLEANING DATASET (Injecting Video IDs & Removing Duplicates)...")
    
    files = glob.glob(f"{MASTER_DIR}/*.csv")
    files = [f for f in files if "clean_master_dataset" not in f and "augmented" not in f]
    
    if not files:
        print("❌ No data files found.")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty: continue
            
            # CRITICAL FIX: Infer Video ID from filename if missing
            # Filename format: fixed_tracks_UUID.csv or just UUID.csv
            # If the file was created by train_master_brain, it might just be UUID.csv
            fname = Path(f).stem
            if fname.startswith("fixed_tracks_"):
                job_id = fname.replace("fixed_tracks_", "")
            else:
                job_id = fname
            
            # Force overwrite/add video_id
            df['video_id'] = job_id
                
            # Ensure frame is int
            if 'frame' in df.columns:
                df['frame'] = df['frame'].fillna(-1).astype(int)
                
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Skipped {f}: {e}")
            
    if not dfs:
        print("❌ No valid data to merge.")
        return

    # Merge
    full_df = pd.concat(dfs, ignore_index=True)
    original_count = len(full_df)
    
    # De-Duplicate
    # Must have video_id now
    if 'video_id' in full_df.columns and 'frame' in full_df.columns:
        clean_df = full_df.drop_duplicates(subset=['video_id', 'frame'], keep='last')
    else:
        print("⚠️ Columns missing after merge. Falling back to simple dedup.")
        clean_df = full_df.drop_duplicates()

    removed = original_count - len(clean_df)
    
    # Save
    clean_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"   📊 Original Count: {original_count}")
    print(f"   🗑️ Removed Duplicates: {removed}")
    print(f"   ✅ UNIQUE SAMPLES: {len(clean_df)}")
    print(f"   💾 Saved Clean Database to: {OUTPUT_FILE}")
    
    # Verify Video ID exists
    if 'video_id' in clean_df.columns:
        print("   ✅ 'video_id' column verified.")
    else:
        print("   ❌ 'video_id' column STILL MISSING.")

if __name__ == "__main__":
    main()
