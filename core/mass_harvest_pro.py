import os
import json
import pandas as pd
import subprocess
from pathlib import Path
from supabase import create_client, Client

# IMPORTS
import sys
sys.path.append(os.getcwd())
from core.event_features_v2 import build_per_frame_base_features

# CONFIG
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Folders
MASTER_BANK = Path("datasets/master_bank")
MASTER_BANK.mkdir(exist_ok=True, parents=True)
TMP_DIR = Path("tmp_harvest")
TMP_DIR.mkdir(exist_ok=True)
FIXED_TRACKS_DIR = Path("runs/json") # Overwrite the bad files here

def process_job(job):
    job_id = job['id']
    video_url = job['source_video_url']
    local_video = TMP_DIR / f"{job_id}.mp4"
    
    # The destination for the "Good" tracks
    final_tracks_path = FIXED_TRACKS_DIR / f"fixed_tracks_{job_id}.json"
    
    print(f"\n🏭 PROCESSING JOB: {job_id}")

    # 1. Download Video
    if not local_video.exists():
        print("   ⬇️ Downloading video...")
        try:
            data = supabase.storage.from_("match-videos").download(video_url)
            local_video.write_bytes(data)
        except:
            print("   ⚠️ Failed to download. Skipping.")
            return

    # 2. Run Tracking (Fresh)
    # We use the 'auto' colors by default since we want scale
    print("   🎥 Running Computer Vision (Tracking)...")
    
    try:
        subprocess.run([
            "python", "core/run_tracker_cli.py",
            "--input", str(local_video),
            "--save", str(final_tracks_path)
        ], check=True)
    except subprocess.CalledProcessError:
        print("   ❌ Tracking failed.")
        return

    # Verify Size
    size_mb = final_tracks_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Tracks Generated: {size_mb:.2f} MB")
    
    if size_mb < 2.0:
        print("   ⚠️ WARNING: File too small. Likely truncated.")

    # 3. Run Auto-Labeling (Miner)
    print("   ⛏️ Mining Labels...")
    subprocess.run([
        "python", "core/events_from_tracks_pipeline.py",
        "--input", str(final_tracks_path)
    ], check=True)
    
    # 4. Extract Physics Features
    print("   🧠 Extracting Physics Vectors...")
    try:
        feats, fps, _ = build_per_frame_base_features(str(final_tracks_path))
        print(f"   📊 Extracted features for {len(feats)} frames.")
        
        # 5. Merge Labels & Features
        labels_path = "runs/json/silver_labels.json" # Miner output
        if os.path.exists(labels_path):
            labels = json.loads(Path(labels_path).read_text())
            
            rows = []
            for evt in labels:
                idx = evt['frame']
                if idx < len(feats):
                    row = {f"f_{k}": v for k,v in enumerate(feats[idx])}
                    row['label'] = evt['label']
                    row['frame'] = idx
                    rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                csv_path = MASTER_BANK / f"{job_id}.csv"
                df.to_csv(csv_path, index=False)
                print(f"   💾 SAVED TO BANK: {csv_path} ({len(df)} events)")
            else:
                print("   ⚠️ No events found to save.")
    except Exception as e:
        print(f"   ❌ Feature extraction failed: {e}")

    # Cleanup video (keep tracks)
    if local_video.exists(): local_video.unlink()

def main():
    # Fetch last 10 successful jobs
    res = supabase.table("ml_jobs").select("*").eq("status", "done").order("created_at", desc=True).limit(10).execute()
    print(f"🚀 Starting Deep Harvest on {len(res.data)} videos...")
    
    for job in res.data:
        process_job(job)

if __name__ == "__main__":
    main()
