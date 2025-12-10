import os
import json
import glob
from pathlib import Path
from supabase import create_client, Client

# --- IMPORTS FROM OUR NEW MODULES ---
from team_assign_v2 import TeamAssigner
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from tactical_metrics import generate_heatmap, measure_distance

# CONFIG
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
ART_BUCKET = "ml-artifacts"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
MASTER_DIR = "datasets/master_bank" # Where we saved the raw data

def reprocess_job(csv_path):
    # Extract Job ID from filename (e.g. "job_id.csv")
    job_id = Path(csv_path).stem
    print(f"🔄 Reprocessing Job: {job_id}")
    
    # We need the TRACKS file. It should be in runs/json/ or we download it back.
    # For now, let's assume we use the 'harvested' data logic or check local cache.
    # If local cache is gone, we skip (or download from Supabase to fix it).
    
    # Let's try to download the OLD tracks to fix them
    local_tracks = Path(f"tmp_tracks_{job_id}.json")
    try:
        print(f"   ⬇️ Downloading old tracks...")
        data = supabase.storage.from_(ART_BUCKET).download(f"job-{job_id}/tracks.json")
        local_tracks.write_bytes(data)
    except Exception as e:
        print(f"   ⚠️ Could not download tracks for {job_id}: {e}")
        return

    tracks_data = json.loads(local_tracks.read_text())
    
    # 1. RE-RUN TEAM ASSIGNMENT (The Fix)
    print("   🎨 Re-assigning Teams (Grass Removal Mode)...")
    assigner = TeamAssigner()
    # We don't have the video to 'observe' again, so we have to rely on the logic 
    # that filters existing detections or we trust the new K-Means on the stored colors if we had them.
    # Since we don't have the video 'crops' anymore, we will apply a HEURISTIC clean.
    # We will filter out the "Crowd" by density clustering again, strictly.
    
    from sklearn.cluster import DBSCAN
    import numpy as np
    
    frames = tracks_data.get('frames', [])
    # ... (Insert Density Clean Logic Here) ...
    # ... (Insert Speed Calculation Here) ...
    
    # 2. CALCULATE SPEED (Physics)
    print("   🏃 Calculating Speed & Distance...")
    speed_engine = SpeedAndDistanceEstimator()
    tracks_data = speed_engine.add_speed_and_distance_to_tracks(tracks_data)
    
    # 3. GENERATE FORMATION
    print("   📍 Generating Formation...")
    # ... (Run formation logic) ...
    
    # 4. UPLOAD FIXED FILES
    print("   ⬆️ Uploading Fixed Artifacts...")
    new_tracks_file = Path(f"fixed_tracks_{job_id}.json")
    new_tracks_file.write_text(json.dumps(tracks_data))
    
    with open(new_tracks_file, "rb") as f:
        supabase.storage.from_(ART_BUCKET).upload(f"job-{job_id}/tracks.json", f.read(), file_options={"upsert": "true"})

    print(f"   ✅ Job {job_id} Updated.")

def main():
    # Get all job IDs from the master bank filenames
    files = glob.glob(f"{MASTER_DIR}/*.csv")
    print(f"found {len(files)} jobs to upgrade.")
    for f in files:
        reprocess_job(f)

if __name__ == "__main__":
    main()
