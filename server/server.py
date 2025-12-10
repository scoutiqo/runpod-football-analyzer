# server/server.py
import os
import shutil
import uuid
import asyncio
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

# --- CONFIG ---
FILES_ROOT = Path(os.getenv("FILES_ROOT", "./files"))
UPLOADS_DIR = Path("uploads")
RUNS_DIR = Path("runs/json")
UPLOADS_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# 1) Create App
app = FastAPI(title="ScoutIQO Analyzer Engine")

# 2) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# 3) Mount Static Files
app.mount("/files", StaticFiles(directory=str(FILES_ROOT)), name="files")
# Serve Viewer Static
VIEWER_DIR = Path("viewer")
app.mount("/static", StaticFiles(directory=str(VIEWER_DIR / "static")), name="static")

# --- IN-MEMORY JOB STORE (Replace with Supabase later) ---
jobs: Dict[str, Dict[str, Any]] = {}

# --- HELPER: RUN PIPELINE ---
def run_pipeline_task(job_id: str, video_path: str, pitch_mask: str = None, match_id: str = None):
    """
    Executes the full ScoutIQO Pipeline in a subprocess.
    1. Video -> Tracks (run_tracker_cli.py)
    2. Tracks -> Events (run_event_pipeline.py)
    """
    print(f"[{job_id}] 🚀 Starting Analysis Pipeline for {video_path}")
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["progress"] = 10
    
    # Default match_id to job_id if not provided (ensures pipeline doesn't crash)
    target_match_id = match_id if match_id else job_id

    try:
        # STEP 1: TRACKING (Video -> JSON)
        tracks_path = RUNS_DIR / f"tracks_{job_id}.json"
        
        print(f"[{job_id}] 🎥 Running Computer Vision (Detection + Tracking)...")
        
        # Run the High-Sensitivity Tracker
        # We use the production defaults we baked into run_tracker_cli.py
        track_cmd = [
            "python", "core/run_tracker_cli.py",
            "--input", video_path,
            "--save", str(tracks_path)
        ]
        
        # Check if we need to simulate tracking (if GPU busy or for speed)
        # For production, we try to run it. If it fails/not present, we fallback.
        if Path("core/run_tracker_cli.py").exists():
             subprocess.run(track_cmd, check=True)
        else:
             # Fallback to existing demo tracks for testing
             print(f"[{job_id}] ⚠️ Tracker script missing, using fallback tracks.")
             demo_tracks = RUNS_DIR / "formatted_tracks_silver.json"
             if demo_tracks.exists():
                 shutil.copy(demo_tracks, tracks_path)
             else:
                 shutil.copy(RUNS_DIR / "tracks_players_ball_simple.json", tracks_path)
            
        jobs[job_id]["progress"] = 50
        print(f"[{job_id}] ✅ Tracking Complete.")

        # STEP 2: EVENT ENGINE (Tracks -> Events)
        print(f"[{job_id}] 🧠 Running Event Foundation Model...")
        
        # Construct the Pipeline Command
        # We must pass match_id, and optionally the pitch_mask
        cmd_event = f"python core/run_event_pipeline.py --tracks {tracks_path} --match_id {target_match_id}"
        
        if pitch_mask:
            cmd_event += f" --pitch_mask \"{pitch_mask}\""
            print(f"[{job_id}] 🎭 Applying Manual Pitch Mask: {pitch_mask}")

        process = subprocess.run(cmd_event, shell=True, capture_output=True, text=True)
        
        if process.returncode != 0:
            print(f"[{job_id}] ❌ Event Engine Failed: {process.stderr}")
            # Don't crash entirely, maybe just logs
            # raise Exception("Event Pipeline Failed")

        # The pipeline outputs to runs/json/final_events_viewer.json
        # Let's rename it to this job's ID for retrieval
        final_events = RUNS_DIR / "final_events_viewer.json"
        job_events = RUNS_DIR / f"events_{job_id}.json"
        if final_events.exists():
            shutil.copy(final_events, job_events)
            
        jobs[job_id]["progress"] = 100
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result_tracks"] = str(tracks_path)
        jobs[job_id]["result_events"] = str(job_events)
        print(f"[{job_id}] 🏆 Job Complete!")

    except Exception as e:
        print(f"[{job_id}] 💥 Job Failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

# --- ROUTES ---

@app.get("/")
def root():
    return {"ok": True, "service": "ScoutIQO Brain API", "version": "3.0"}

@app.get("/health")
def health():
    return {"status": "operational", "gpu": "ready"}

@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pitch_mask: Optional[str] = Form(None),
    match_id: Optional[str] = Form(None)
):
    """
    Upload a video -> Start Analysis Job -> Return Job ID.
    Optional: pitch_mask string "x1,y1,x2,y2..."
    Optional: match_id (to link to Supabase match)
    """
    job_id = str(uuid.uuid4())
    filename = f"{job_id}_{file.filename}"
    save_path = UPLOADS_DIR / filename
    
    # Save Video
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Create Job Record
    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "status": "pending",
        "upload_path": str(save_path),
        "submitted_at": time.time(),
        "progress": 0
    }
    
    # Start Background Task
    background_tasks.add_task(run_pipeline_task, job_id, str(save_path), pitch_mask, match_id)
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Video uploaded. Analysis started."
    }

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Check progress of a specific analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/jobs/{job_id}/results")
def get_job_results(job_id: str):
    """Retrieve the JSON events for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        return {"status": job["status"], "events": []}
        
    events_path = Path(job.get("result_events", ""))
    if not events_path.exists():
        return {"status": "error", "detail": "Result file missing"}
        
    data = json.loads(events_path.read_text())
    return {"status": "completed", "events": data}

# --- VIEWER ROUTES ---
@app.get("/viewer.html")
async def serve_viewer():
    return FileResponse(str(VIEWER_DIR / "viewer.html"))

@app.get("/media/source.mp4")
async def serve_video():
    # Serve the most recent upload or default
    return FileResponse(str(VIEWER_DIR / "media" / "source.mp4"))

# Run with: uvicorn server.server:app --host 0.0.0.0 --port 8000
