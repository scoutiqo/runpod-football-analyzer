import json
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# CONFIG
SYLLABUS_FILE = "datasets/master_bank/oracle_syllabus_deep.json"
CSV_FILE = "datasets/master_bank/clean_master_dataset.csv"
TRACKS_DIR = "runs/json"

# Import feature extractor to rebuild vectors
sys.path.append(os.getcwd())
from core.event_features_v2 import build_per_frame_base_features, build_event_feature_vector

def main():
    print("🧠 RESTORING MEMORY (Injecting Syllabus into CSV)...")
    
    if not Path(SYLLABUS_FILE).exists():
        print("❌ Syllabus not found.")
        return

    # 1. Load Syllabus
    syllabus = json.loads(Path(SYLLABUS_FILE).read_text())
    print(f"   📚 Found {len(syllabus)} concepts in Syllabus.")
    
    # 2. Load Existing CSV
    if Path(CSV_FILE).exists():
        df = pd.read_csv(CSV_FILE)
        print(f"   📄 Existing CSV has {len(df)} rows.")
    else:
        df = pd.DataFrame()
        
    new_rows = []
    processed_videos = {}
    
    # 3. Process Syllabus
    for item in syllabus:
        label = item.get('label')
        frame_idx = int(item.get('frame'))
        video_id = item.get('video')
        
        if not video_id or label in ["unknown", "none"]: continue
        
        # Check if already in DF
        if not df.empty and 'video_id' in df.columns:
            exists = ((df['video_id'] == video_id) & (df['frame'] == frame_idx)).any()
            if exists: continue # Skip duplicates
            
        # Feature Extraction
        track_path = f"{TRACKS_DIR}/fixed_tracks_{video_id}.json"
        if not os.path.exists(track_path): 
            # Try finding it in tmp_jobs or just skip if track file lost
            continue

        if video_id not in processed_videos:
            try:
                feats, fps, _ = build_per_frame_base_features(track_path)
                processed_videos[video_id] = (feats, fps)
            except:
                processed_videos[video_id] = None
        
        if processed_videos[video_id]:
            feats, fps = processed_videos[video_id]
            if frame_idx < len(feats):
                vec = build_event_feature_vector(frame_idx, feats, fps=fps)
                row = {f"f_{k}": v for k, v in enumerate(vec)}
                row['label'] = label
                row['frame'] = frame_idx
                row['video_id'] = video_id
                new_rows.append(row)

    # 4. Merge and Save
    if new_rows:
        print(f"   💉 Injecting {len(new_rows)} recovered memories...")
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([df, new_df], ignore_index=True)
        
        # Final Dedup
        if 'video_id' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['video_id', 'frame'], keep='last')
            
        combined_df.to_csv(CSV_FILE, index=False)
        print(f"   ✅ Memory Restored. CSV now has {len(combined_df)} samples.")
        print(f"   💾 Saved to {CSV_FILE}")
    else:
        print("   ⚠️ No new memories injected (Files might be missing or already up to date).")

if __name__ == "__main__":
    main()
