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

MASTER_BANK = Path("datasets/master_bank")
MASTER_BANK.mkdir(exist_ok=True, parents=True)
TMP_DIR = Path("tmp_harvest")
TMP_DIR.mkdir(exist_ok=True)

def process_job(job):
    job_id = job['id']
    video_url = job['source_video_url']
    local_video = TMP_DIR / f"{job_id}.mp4"
    local_tracks = TMP_DIR / f"tracks_{job_id}.json"
    
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
    print("   🎥 Running Computer Vision (Tracking)...")
    # Using Auto-Colors by default
    subprocess.run([
        "python", "core/run_tracker_cli.py",
        "--input", str(local_video),
        "--save", str(local_tracks)
    ], check=True)

    # 3. Run Auto-Labeling (Heuristics for Training Data)
    # We use the Miner to generate "Silver Labels" which the LSTM will learn to mimic
    print("   ⛏️ Mining Labels...")
    subprocess.run([
        "python", "core/events_from_tracks_pipeline.py",
        "--input", str(local_tracks)
    ], check=True)
    
    # 4. Extract Physics Features
    print("   🧠 Extracting Physics Vectors...")
    feats, fps, _ = build_per_frame_base_features(str(local_tracks))
    
    # 5. Merge Labels & Features into CSV
    labels_path = "runs/json/silver_labels.json" # Default output of miner
    if os.path.exists(labels_path):
        labels = json.loads(Path(labels_path).read_text())
        
        rows = []
        for evt in labels:
            idx = evt['frame']
            if idx < len(feats):
                row = {f"f_{k}": v for k,v in enumerate(feats[idx])}
                row['label'] = evt['label']
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = MASTER_BANK / f"{job_id}.csv"
            df.to_csv(csv_path, index=False)
            print(f"   ✅ SAVED DATASET: {csv_path} ({len(df)} samples)")
        else:
            print("   ⚠️ No events found.")

    # Cleanup
    if local_video.exists(): local_video.unlink()
    if local_tracks.exists(): local_tracks.unlink()

def main():
    # Fetch last 10 successful jobs
    res = supabase.table("ml_jobs").select("*").eq("status", "done").order("created_at", desc=True).limit(10).execute()
    print(f"🚀 Starting Factory Run on {len(res.data)} videos...")
    
    for job in res.data:
        process_job(job)

if __name__ == "__main__":
    main()
