import os
import json
import sys
from pathlib import Path
from supabase import create_client

# CONFIG
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_DIR = Path("/workspace/runpod-football-analyzer/runs/json")
ARTIFACT_BUCKET = "ml-artifacts"

# HARDCODED ID from your logs
JOB_ID = "c3d42f40-8e8a-4b17-946d-02c1974da622"

def main():
    print(f"🚀 MANUALLY UPLOADING RESULTS for Job {JOB_ID}...")

    # Files to upload
    files_to_upload = {
        "events.json": BASE_DIR / "final_events_viewer.json",
        "chains.json": BASE_DIR / "possession_chains.json",
        "formation.json": BASE_DIR / "formation.json",
        "tactics.json": BASE_DIR / "advanced_metrics.json",
        # Add tracks if you want to re-upload them (large)
        # "tracks.json": BASE_DIR / "tracks.json" 
    }
    
    paths = {}
    
    for remote_name, local_path in files_to_upload.items():
        if local_path.exists():
            key = f"job-{JOB_ID}/{remote_name}"
            print(f"   - Uploading {remote_name}...")
            try:
                with open(local_path, "rb") as f:
                    supabase.storage.from_(ARTIFACT_BUCKET).upload(key, f.read(), file_options={"upsert": "true"})
                paths[f"{remote_name.split('.')[0]}_url"] = key
            except Exception as e:
                print(f"     ⚠️ Upload failed: {e}")
        else:
            print(f"   ⚠️ File not found: {local_path}")

    # Update Database
    if paths:
        print("   💾 Updating Database Record...")
        # Get existing artifacts first
        res = supabase.table("ml_jobs").select("artifacts").eq("id", JOB_ID).execute()
        current_artifacts = res.data[0]['artifacts'] if res.data and res.data[0]['artifacts'] else {}
        
        # Merge
        current_artifacts.update(paths)
        
        supabase.table("ml_jobs").update({
            "artifacts": current_artifacts,
            "status": "done"
        }).eq("id", JOB_ID).execute()
        
        print("✅ Database updated successfully.")
    else:
        print("❌ No files were uploaded.")

if __name__ == "__main__":
    main()
