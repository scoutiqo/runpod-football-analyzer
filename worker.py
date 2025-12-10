import os
import time
import json
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path
from supabase import create_client, Client
from data_manager import StorageManager  # <--- NEW: Connects to Backblaze

# --- CONFIG ---
# We still use Supabase DB for tracking status, but NOT for file storage
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
WORKER_ID = os.getenv("ML_WORKER_ID", "vast-gpu-1")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("⚠️ WARNING: Supabase keys missing. Database updates will be skipped.")
    supabase = None

# Initialize Backblaze Manager
storage = StorageManager()

BASE_DIR = Path("/workspace/runpod-football-analyzer")
TMP_DIR = BASE_DIR / "tmp_jobs"
TMP_DIR.mkdir(exist_ok=True)

def download_video_from_backblaze(filename):
    """
    Downloads the specific video file from Backblaze uploads/ folder.
    """
    local_path = TMP_DIR / filename
    cloud_key = f"uploads/{filename}"
    
    print(f"[{WORKER_ID}] ⬇️ Downloading {cloud_key} from Backblaze...")
    
    if storage.download_file(cloud_key, str(local_path)):
        if local_path.stat().st_size > 1000: # Check if not empty
            print(f"[{WORKER_ID}] ✅ Downloaded: {filename} ({local_path.stat().st_size / 1024 / 1024:.2f} MB)")
            return local_path
            
    print(f"[{WORKER_ID}] ❌ Download failed.")
    return None

def run_pipeline(video_path, job_id):
    """
    Your Original PRO Pipeline logic. 
    """
    print(f"[{WORKER_ID}] 🚀 Running PRO Pipeline on {video_path.name}...")
    tracks_final = BASE_DIR / "runs" / "json" / "tracks.json"

    # 1. Run the Main Tracker (YOLO + Calibration)
    cmd = ["python", "core/run_pro_pipeline.py", "--video", str(video_path), "--match_id", job_id, "--save_tracks", str(tracks_final)]
    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)

    # 2. Run the Specialist Modules
    print(f"[{WORKER_ID}] 🎽 Jersey OCR...")
    subprocess.run(["python", "core/apply_jersey_ocr.py"], cwd=str(BASE_DIR), check=True)

    print(f"[{WORKER_ID}] 🧵 Stitching Tracks...")
    subprocess.run(["python", "core/track_stitcher.py"], cwd=str(BASE_DIR), check=True)

    print(f"[{WORKER_ID}] ⛏️ Event Mining...")
    subprocess.run(["python", "core/events_from_tracks_pipeline.py", "--input", str(tracks_final)], cwd=str(BASE_DIR), check=True)

    # Note: Skipping build_event_dataset if training isn't needed every run, 
    # but keeping LSTM inference which is critical.
    
    print(f"[{WORKER_ID}] 🧠 LSTM Inference (Tactical Brain)...")
    subprocess.run(["python", "core/predict_events_lstm.py", "--tracks", str(tracks_final), "--output", "runs/json/predicted_events_learned.json"], cwd=str(BASE_DIR), check=True)

    print(f"[{WORKER_ID}] 📦 Exporting Events...")
    subprocess.run(["python", "core/export_events_for_viewer.py"], cwd=str(BASE_DIR), check=True)

    print(f"[{WORKER_ID}] 📈 xG/xT Metrics...")
    subprocess.run(["python", "core/value_metrics.py"], cwd=str(BASE_DIR), check=True)

    print(f"[{WORKER_ID}] 📊 Tactical Metrics...")
    subprocess.run(["python", "core/tactical_metrics.py"], cwd=str(BASE_DIR), check=True)

    print(f"[{WORKER_ID}] 📍 Formation Analysis...")
    subprocess.run(["python", "core/analyze_formation.py"], cwd=str(BASE_DIR), check=True)

    print(f"[{WORKER_ID}] 🔗 Possession Chains...")
    subprocess.run(["python", "core/possession_engine.py"], cwd=str(BASE_DIR), check=True)

    return tracks_final

def upload_results_to_backblaze(job_id):
    """
    Uploads all generated JSONs to Backblaze 'processed/' folder.
    """
    print(f"[{WORKER_ID}] 📦 Uploading Artifacts to Backblaze...")

    # Run finalizers
    subprocess.run(["python", "core/clean_json_for_upload.py"], cwd=str(BASE_DIR), check=True)
    subprocess.run(["python", "core/finalize_export.py"], cwd=str(BASE_DIR), check=True)

    files_map = {
        "meta.json": BASE_DIR / "runs/json/meta.json",
        "tracks.json": BASE_DIR / "runs/json/tracks_vis.json",
        "events.json": BASE_DIR / "runs/json/final_events_viewer.json",
        "formation.json": BASE_DIR / "runs/json/formation.json",
        "tactics.json": BASE_DIR / "runs/json/advanced_metrics.json",
        "chains.json": BASE_DIR / "runs/json/possession_chains.json",
        "colors.json": BASE_DIR / "runs/json/team_colors.json"
    }

    uploaded_paths = {}
    
    for remote_name, local_path in files_map.items():
        if local_path.exists():
            cloud_key = f"processed/{job_id}/{remote_name}"
            print(f"   - Uploading {remote_name}...")
            storage.upload_file(str(local_path), cloud_key)
            uploaded_paths[remote_name] = cloud_key

    return uploaded_paths

def main_loop():
    print(f"[{WORKER_ID}] 🚀 Autonomous Worker Started (Backblaze Mode)")
    print(f"[{WORKER_ID}] 👀 Watching {storage.bucket}/uploads/...")

    while True:
        try:
            # 1. POLL BACKBLAZE directly (No DB required to start)
            new_files = storage.list_new_videos()
            
            if not new_files:
                print("💤 No pending videos. Sleeping 30s...")
                time.sleep(30)
                continue

            # 2. CLAIM JOB
            cloud_key = new_files[0] # e.g. "uploads/match_123.mp4"
            filename = os.path.basename(cloud_key)
            job_id = filename.split('.')[0] # e.g. "match_123"
            
            print(f"\n⚡ DETECTED JOB: {job_id}")

            # 3. DOWNLOAD
            video_path = download_video_from_backblaze(filename)
            if not video_path:
                time.sleep(10)
                continue

            # 4. RUN FULL AI PIPELINE
            run_pipeline(video_path, job_id)

            # 5. UPLOAD RESULTS
            artifacts = upload_results_to_backblaze(job_id)

            # 6. UPDATE DB (If connected)
            if supabase:
                try:
                    supabase.table("ml_jobs").upsert({
                        "id": job_id,
                        "status": "done",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "artifacts": artifacts
                    }).execute()
                    print(f"[{WORKER_ID}] ✅ Database Updated.")
                except Exception as e:
                    print(f"[{WORKER_ID}] ⚠️ Database Update Failed (but files are safe): {e}")

            # 7. CLEANUP
            print(f"[{WORKER_ID}] 🧹 Archiving raw video...")
            storage.move_to_processed(cloud_key)
            
            # Delete local file
            if video_path.exists(): os.remove(video_path)
            
            print(f"[{WORKER_ID}] ✨ Job Complete.\n")

        except Exception as e:
            print(f"[{WORKER_ID}] ❌ CRITICAL ERROR: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main_loop()
