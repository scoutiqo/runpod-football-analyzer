import os
import uuid
import json
import subprocess
from pathlib import Path
from typing import Optional, Any, Dict

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parents[1]  # /workspace/runpod-football-analyzer
UPLOADS_DIR = ROOT / "uploads" / "api_uploads"
RUNS_DIR = ROOT / "runs" / "api_jobs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    print("[video_api] Supabase REST configured")
else:
    print("[video_api] Supabase env vars not set; will skip inserts")


app = FastAPI(title="ScoutIQO Video Analyzer API")

# CORS – you can tighten allow_origins later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


def run_analyze(video_path: Path, out_dir: Path) -> None:
    """
    Call the existing CLI:
        python analyze.py --video <video_path> --out <out_dir>
    """
    cmd = [
        "python",
        "analyze.py",
        "--video",
        str(video_path),
        "--out",
        str(out_dir),
    ]
    print(f"[video_api] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("[video_api] analyze.py stderr:\n", result.stderr)
        raise RuntimeError(f"analyze.py failed with code {result.returncode}")
    print("[video_api] analyze.py finished successfully")


def safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def summarize_events(events: Any) -> Dict[str, Any]:
    """
    Very defensive summary: count labels if events is a list of dicts.
    """
    summary: Dict[str, Any] = {"total_events": 0, "label_counts": {}}
    if isinstance(events, list):
        summary["total_events"] = len(events)
        counts: Dict[str, int] = {}
        for ev in events:
            if isinstance(ev, dict):
                label = ev.get("label") or ev.get("event_type") or ev.get("type")
                if label:
                    counts[label] = counts.get(label, 0) + 1
        summary["label_counts"] = counts
    return summary


def supabase_insert_video_analysis(row: Dict[str, Any]) -> Optional[Any]:
    """
    Insert a row into public.video_analyses via Supabase REST.
    Returns the inserted row(s) or None on failure.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("[video_api] Supabase env not set; skipping insert")
        return None

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/video_analyses"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(row), timeout=15)
    except Exception as e:
        print("[video_api] Supabase insert exception:", e)
        return None

    if not resp.ok:
        print(
            "[video_api] Supabase insert failed:",
            resp.status_code,
            resp.text[:500],
        )
        return None

    try:
        return resp.json()
    except Exception:
        return None


@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
):
    """
    Accepts a video file, runs analyze.py, optionally logs to Supabase,
    and returns JSON with paths + basic summary.
    """
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    job_id = str(uuid.uuid4())
    out_dir = RUNS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    video_filename = f"{job_id}_{file.filename}"
    video_path = UPLOADS_DIR / video_filename

    # Save uploaded file to disk
    try:
        contents = await file.read()
        with video_path.open("wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # Run the ML pipeline
    try:
        run_analyze(video_path, out_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analyze.py failed: {e}")

    # Collect outputs
    tracks_path = out_dir / "tracks.json"
    events_path = out_dir / "events.json"
    metadata_path = out_dir / "metadata.txt"

    events = safe_load_json(events_path)
    tracks = safe_load_json(tracks_path)
    summary = summarize_events(events)

    # Optional: save to Supabase via REST
    supabase_row = None
    row: Dict[str, Any] = {
        "video_filename": file.filename,
        "video_path": str(video_path),
        "tracks_path": str(tracks_path),
        "events_path": str(events_path),
        "summary": summary,
    }
    if user_id:
        row["user_id"] = user_id

    inserted = supabase_insert_video_analysis(row)
    if inserted is not None:
        supabase_row = inserted

    return {
        "job_id": job_id,
        "video_filename": file.filename,
        "video_path": str(video_path),
        "tracks_path": str(tracks_path),
        "events_path": str(events_path),
        "summary": summary,
        "supabase_row": supabase_row,
    }
