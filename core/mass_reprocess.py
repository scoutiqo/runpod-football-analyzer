import os
import json
import time
import subprocess
from pathlib import Path
from supabase import create_client, Client

# CONFIG
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
RAW_BUCKET = "match-videos"
BASE_DIR = Path("/workspace/runpod-football-analyzer")
TMP_DIR = BASE_DIR / "tmp_reprocess"
TMP_DIR.mkdir(exist_ok=True)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def reprocess_job(job):
    job_id = job['id']
    video_path_remote = job['source_video_url']
    local_video = TMP_DIR / f"{job_id}.mp4"
    
    print(f"\n🔄 REPROCESSING JOB: {job_id}")
    
    # 1. Download Video
    if not local_video.exists():
        print(f"   ⬇️ Downloading video...")
        try:
            data = supabase.storage.from_(RAW_BUCKET).download(video_path_remote)
            local_video.write_bytes(data)
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            return

    # 2. Track (Generate Fresh Data)
    print(f"   🎥 Tracking (This takes time)...")
    raw_tracks_file = BASE_DIR / f"raw_tracks_{job_id}.json"
    
    # Using the Color/Mask args from the job if available
    artifacts = job.get('artifacts') or {}
    mask = job.get('perspective') or artifacts.get('pitch_mask')
    col_a = artifacts.get('team_a_color')
    col_b = artifacts.get('team_b_color')
    
    cmd_track = [
        "python", "core/run_tracker_cli.py",
        "--input", str(local_video),
        "--save", str(raw_tracks_file)
    ]
    if col_a and col_b: cmd_track.extend(["--color_a", col_a, "--color_b", col_b])
    
    subprocess.run(cmd_track, cwd=str(BASE_DIR), check=True)

    # 3. Pipeline (CONNECT THE DOTS)
    # Crucial: We pass the FRESH tracks file explicitly
    print(f"   🧠 Running Event Engine on FRESH data...")
    cmd_pipeline = [
        "python", "core/run_event_pipeline.py",
        "--match_id", job_id,
        "--tracks", str(raw_tracks_file), # <--- THE FIX
        "--video", str(local_video)
    ]
    if mask: cmd_pipeline.extend(["--pitch_mask", mask])
    
    subprocess.run(cmd_pipeline, cwd=str(BASE_DIR), check=True)
    
    print(f"   ✅ Job {job_id} Fixed & Harvested.")
    
    # Cleanup to save space
    if local_video.exists(): local_video.unlink()
    if raw_tracks_file.exists(): raw_tracks_file.unlink()

def main():
    # Fetch the last 10 'done' jobs
    res = supabase.table("ml_jobs").select("*").eq("status", "done").order("created_at", desc=True).limit(10).execute()
    jobs = res.data
    
    print(f"🚀 Found {len(jobs)} jobs to re-process correctly.")
    for job in jobs:
        reprocess_job(job)

if __name__ == "__main__":
    main()
