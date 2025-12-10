#!/usr/bin/env python3
"""
Simplified ScoutIQO Analyzer Server
Avoids problematic imports and provides basic functionality
"""

import os, uuid, json, tempfile, subprocess, time, logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from fastapi import Form

# --- .env support (optional) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------- Config --------
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "").strip()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "scoutiqo")
CALLBACK_SECRET = os.getenv("CALLBACK_SECRET", "")

app = FastAPI(title="ScoutIQO Analyzer", docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("uvicorn")

# -------- static media root --------
MEDIA_ROOT = Path("data")
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_ROOT)), name="media")

# -------- simple WS pub/sub --------
channels: Dict[str, List[WebSocket]] = {}
async def publish(job_id: str, event: dict):
    for ws in list(channels.get(job_id, [])):
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            try:
                channels[job_id].remove(ws)
            except Exception:
                pass

# -------- persistent storage --------
import pickle
import atexit

# File paths for persistence
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = STORAGE_DIR / "progress.pkl"
JOBS_FILE = STORAGE_DIR / "jobs.pkl"

# Load existing data
def load_storage():
    progress_store = {}
    job_storage = {}
    
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'rb') as f:
                progress_store = pickle.load(f)
        except:
            pass
    
    if JOBS_FILE.exists():
        try:
            with open(JOBS_FILE, 'rb') as f:
                job_storage = pickle.load(f)
        except:
            pass
    
    return progress_store, job_storage

# Save data
def save_storage():
    try:
        with open(PROGRESS_FILE, 'wb') as f:
            pickle.dump(progress_store, f)
        with open(JOBS_FILE, 'wb') as f:
            pickle.dump(job_storage, f)
    except Exception as e:
        log.error("Failed to save storage: %s", e)

# Load initial data
progress_store, job_storage = load_storage()

# Register cleanup function
atexit.register(save_storage)

# -------- ffmpeg helpers --------
def _run_ffmpeg(cmd: list):
    log.info("FFmpeg: %s", " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        out = p.stdout.decode(errors="ignore")
        log.error("FFmpeg failed:\n%s", out[:2000])
        raise RuntimeError(out)

def segment_copy_only_local(local_input: str, seg_seconds: int, job_id: str) -> list[Path]:
    """Copy-only segmentation into a temporary directory."""
    out_dir = Path(tempfile.mkdtemp()) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        "ffmpeg","-y","-i", local_input, "-reset_timestamps","1","-map","0","-c","copy",
        "-segment_time", str(seg_seconds), "-f","segment", str(out_dir / "seg_%03d.mp4")
    ])
    return sorted(out_dir.glob("seg_*.mp4"))

# -------- API --------
@app.get("/")
def root():
    return {"ok": True, "health": "/health", "docs": "/docs", "monitor": "/monitor/<job_id>"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile = File(...), segment_seconds: int = 20, fast: int = 1, simulate: bool = False):
    """Upload and segment video file"""
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_in = tempfile.mktemp(suffix=suffix)
    job_id = f"up_{uuid.uuid4().hex[:8]}"

    # Save uploaded file to temp
    with open(tmp_in, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    # Get video metadata (simplified)
    video_width = 1920
    video_height = 1080
    video_fps = 30.0
    video_duration = 60.0  # Default duration

    # Store job metadata
    job_storage[job_id] = {
        "status": "uploaded",
        "video_metadata": {
            "duration_s": video_duration,
            "width": video_width,
            "height": video_height,
            "fps": video_fps
        },
        "segments": [],
        "created_at": time.time()
    }
    save_storage()  # Persist immediately

    if simulate:
        # Simulate segmentation
        num_segments = max(1, int(video_duration / segment_seconds))
        signed_urls = [f"simulated://segment_{i:03d}.mp4" for i in range(num_segments)]
    else:
        # Segment locally (temp dir)
        seg_paths = segment_copy_only_local(tmp_in, seg_seconds=segment_seconds, job_id=job_id)
        
        # For now, just use local paths (in production, upload to Supabase)
        signed_urls = [str(p) for p in seg_paths]

        # Cleanup temp files (best-effort)
        try:
            Path(tmp_in).unlink(missing_ok=True)
        except Exception:
            pass

    # Update job storage
    job_storage[job_id]["segments"] = signed_urls
    job_storage[job_id]["status"] = "segmented"

    return {"job_id": job_id, "segment_urls": signed_urls}

class AnalyzeReq(BaseModel):
    job_id: str
    segment_urls: List[str]
    preset: str | None = "fast"
    workers: int | None = 4
    simulate: bool = False

@app.post("/analyze")
def analyze(req: AnalyzeReq):
    """Start analysis of video segments"""
    if not req.segment_urls:
        raise HTTPException(400, "segment_urls required")
    
    # Check if job exists
    if req.job_id not in job_storage:
        raise HTTPException(404, f"Job {req.job_id} not found")
    
    # Update job status
    job_storage[req.job_id]["status"] = "analyzing"
    job_storage[req.job_id]["analysis_started"] = time.time()
    
    if req.simulate:
        # Simulate analysis
        job_storage[req.job_id]["simulate"] = True
        return {"job_id": req.job_id, "status": "simulating"}
    
    if not RUNPOD_ENDPOINT:
        raise HTTPException(500, "RUNPOD_ENDPOINT not set")
    
    payload = {
        "job_id": req.job_id,
        "segment_urls": req.segment_urls,
        "callback_url": f"{PUBLIC_BASE_URL}/progress/{req.job_id}",
        "callback_secret": CALLBACK_SECRET,
        "preset": req.preset or "fast",
        "workers": int(req.workers or 4),
        "make_overlay": True,
        "simulate": req.simulate
    }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {RUNPOD_API_KEY}"}
    r = requests.post(RUNPOD_ENDPOINT, headers=headers, json={"input": payload}, timeout=60)
    if r.status_code >= 300:
        raise HTTPException(500, f"RunPod launch failed: {r.text}")
    
    return {"job_id": req.job_id, "status": "started"}

@app.post("/progress/{job_id}")
async def progress(job_id: str, req: Request, x_callback_secret: str | None = Header(None)):
    """Progress callback endpoint"""
    if not CALLBACK_SECRET or x_callback_secret != CALLBACK_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await req.json()
    progress_store.setdefault(job_id, []).append({"ts": time.time(), **data})
    logging.info("PROGRESS %s: %s", job_id, data)
    
    # Update job storage based on progress type
    if job_id in job_storage:
        if data.get("type") == "segment_done":
            job_storage[job_id]["completed_segments"] = job_storage[job_id].get("completed_segments", 0) + 1
        elif data.get("type") == "done":
            job_storage[job_id]["status"] = "completed"
            job_storage[job_id]["completed_at"] = time.time()
            if "tracks_url" in data:
                job_storage[job_id]["tracks_url"] = data["tracks_url"]
            if "auto_tune" in data:
                job_storage[job_id]["auto_tune"] = data["auto_tune"]
        save_storage()  # Persist progress updates
    
    await publish(job_id, data)
    return {"ok": True}

@app.get("/status/{job_id}")
def status(job_id: str):
    evts = progress_store.get(job_id, [])
    return {"job_id": job_id, "events": len(evts), "last_type": (evts[-1]["type"] if evts else None)}

@app.get("/progress/{job_id}/dump")
def dump(job_id: str):
    return {"job_id": job_id, "events": progress_store.get(job_id, [])}

@app.get("/monitor/{job_id}")
def monitor(job_id: str):
    """Monitor job progress with real-time updates"""
    # Check both job_storage and progress_store
    if job_id not in job_storage and job_id not in progress_store:
        raise HTTPException(404, f"Job {job_id} not found")
    
    job_data = job_storage.get(job_id, {
        "status": "unknown",
        "video_metadata": {"duration_s": 0, "width": 0, "height": 0, "fps": 0},
        "segments": [],
        "created_at": time.time()
    })
    return HTMLResponse(f"""
<!doctype html><meta charset="utf-8"/><title>Monitor – {job_id}</title>
<style>
  body{{font-family:system-ui;margin:20px}}
  .row{{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}}
  video{{width:640px;height:360px;background:#000;border-radius:8px}}
  pre{{max-height:520px;overflow:auto;background:#111;color:#eee;padding:12px;border-radius:8px;min-width:420px}}
  .pill{{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef;margin-left:8px}}
  .status{{padding:8px;border-radius:8px;margin:8px 0}}
  .status.uploaded{{background:#e3f2fd}}
  .status.segmented{{background:#f3e5f5}}
  .status.analyzing{{background:#fff3e0}}
  .status.completed{{background:#e8f5e8}}
</style>
<h1>Job <code>{job_id}</code> <span id="badge" class="pill">waiting…</span></h1>
<div class="status status-{job_data.get('status', 'unknown')}">
  Status: <strong>{job_data.get('status', 'unknown')}</strong>
  {f" | Segments: {job_data.get('completed_segments', 0)}/{len(job_data.get('segments', []))}" if job_data.get('segments') else ""}
</div>
<div class="row">
  <div>
    <video id="vid" controls muted playsinline></video>
    <div style="margin-top:8px">
      <button onclick="forcePoll()">Refresh now</button>
      <a id="dump" target="_blank">Open full event dump</a>
    </div>
  </div>
  <pre id="log"></pre>
</div>
<script>
const job  = {json.dumps(job_id)};
const base = location.origin;

const logEl = document.getElementById('log');
const vid   = document.getElementById('vid');
const badge = document.getElementById('badge');
const dumpA = document.getElementById('dump');
dumpA.href  = `${{base}}/progress/${{job}}/dump`;

let lastSeg = -1;
let lastCount = 0;

async function poll() {{
  try {{
    const st = await fetch(`${{base}}/status/${{job}}`).then(r => r.json());
    badge.textContent = st.last_type ? st.last_type : "waiting…";

    if ((st.events || 0) !== lastCount) {{
      lastCount = st.events || 0;

      const dump = await fetch(`${{base}}/progress/${{job}}/dump`).then(r => r.json());
      logEl.textContent = JSON.stringify(dump, null, 2);

      const evts = dump && dump.events ? dump.events : [];
      const latest = evts.length ? evts[evts.length - 1] : {{}};

      if (latest && latest.type === "segment_start" && typeof latest.seg === "number" && latest.url) {{
        if (latest.seg !== lastSeg) {{
          lastSeg = latest.seg;
          vid.src = latest.url;
          try {{ await vid.play(); }} catch (e) {{}}
        }}
      }}
    }}
  }} catch (e) {{
    badge.textContent = "error";
  }}
}}

function forcePoll() {{ lastCount = -1; poll(); }}
setInterval(poll, 1000);
poll();
</script>
""")

@app.get("/files/jobs/{job_id}/tracks.json")
def get_tracks_json(job_id: str):
    """Get tracks.json for a job"""
    if job_id not in job_storage:
        raise HTTPException(404, f"Job {job_id} not found")
    
    job_data = job_storage[job_id]
    
    if job_data.get("status") != "completed":
        raise HTTPException(400, f"Job {job_id} not completed yet")
    
    # Return mock tracks.json data
    return {
        "job_id": job_id,
        "video": job_data.get("video_metadata", {}),
        "calibration": {"homography": None, "units": "px"},
        "players": [],
        "events": [],
        "auto_tune": job_data.get("auto_tune", {}),
        "artifacts": {"overlays": [], "logs": []}
    }

@app.get("/files/jobs/{job_id}/players/tid_{player_id}.json")
def get_player_insights(job_id: str, player_id: int):
    """Get player insights for a specific player"""
    if job_id not in job_storage:
        raise HTTPException(404, f"Job {job_id} not found")
    
    # Return mock player insights
    return {
        "job_id": job_id,
        "tid": player_id,
        "identity": {"name": None, "squad_number": None, "foot": None},
        "minutes": {"on": 0.0, "off": None, "played": 0.0},
        "formations": [],
        "opponent_matchups": [],
        "game_states": [],
        "summary": {
            "touches": 0, "passes_att": 0, "passes_cmp": 0, "xA": 0.0,
            "carries": 0, "prog_carry_px": 0.0, "dribbles_won": 0,
            "shots": 0, "goals": 0, "xG": 0.0, "psxg_on_target": 0.0,
            "pressures": 0, "press_success": 0,
            "duels_air_won_pct": 0.0, "duels_grd_won_pct": 0.0,
            "distance_px": 0.0, "sprints": 0, "max_speed_pxps": 0.0
        },
        "value_models": {"xT_sum": 0.0, "EPV_sum": 0.0, "VAEP_sum": 0.0, "packing_for": 0, "packing_against": 0},
        "role_specific_notes": []
    }

@app.get("/easy")
def easy():
    """Simple HTML page that runs the full pipeline from the browser"""
    return HTMLResponse(f"""
<!doctype html><meta charset="utf-8"/><title>ScoutIQO – Easy Runner</title>
<style>
  body{{font-family:system-ui;background:#0b1320;color:#e9eef6;display:grid;place-items:center;min-height:100vh;margin:0}}
  .card{{width:min(860px,90vw);background:#101a2d;padding:24px;border-radius:16px;box-shadow:0 8px 40px rgba(0,0,0,.35)}}
  h1{{margin:.2rem 0 1rem;font-size:1.4rem}}
  label{{display:block;margin:.7rem 0 .25rem;color:#a9b3c7}}
  input,select{{width:100%;padding:.6rem .75rem;border-radius:10px;border:1px solid #1e2a44;background:#0c1426;color:#e9eef6}}
  .row{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
  button{{margin-top:16px;background:#00c2a8;border:0;color:#04121b;
          padding:.7rem 1rem;border-radius:999px;font-weight:700;cursor:pointer}}
  .hint{{font-size:.9rem;color:#8fa0b8;margin-top:8px}}
  .ok{{display:inline-block;padding:4px 10px;border-radius:999px;background:#123e34;color:#66f0d5;margin-left:8px}}
</style>
<div class="card">
  <h1>ScoutIQO – Easy Runner <span class="ok">BASE: {PUBLIC_BASE_URL}</span></h1>
  <form action="/easy" method="post" enctype="multipart/form-data">
    <label>Video file (.mp4)</label>
    <input type="file" name="file" accept="video/mp4" required>

    <div class="row">
      <div>
        <label>Segment seconds</label>
        <input type="number" name="segment_seconds" value="20" min="4" max="60" />
      </div>
      <div>
        <label>How many segments to analyze</label>
        <input type="number" name="limit_segments" value="3" min="1" max="20" />
      </div>
    </div>

    <div class="row">
      <div>
        <label>Preset</label>
        <select name="preset">
          <option value="fast" selected>fast</option>
          <option value="normal">normal</option>
        </select>
      </div>
      <div>
        <label>Workers</label>
        <input type="number" name="workers" value="2" min="1" max="8" />
      </div>
    </div>

    <div class="row">
      <div>
        <label>Simulate</label>
        <select name="simulate">
          <option value="true" selected>true (quick smoke test)</option>
          <option value="false">false (real run)</option>
        </select>
      </div>
      <div>
        <label>Fast segmentation</label>
        <select name="fast">
          <option value="1" selected>copy-only (fast)</option>
          <option value="0">transcode (safer)</option>
        </select>
      </div>
    </div>

    <button type="submit">Run analysis</button>
    <div class="hint">This page: uploads → segments → uploads to Supabase → launches RunPod → redirects to monitor.</div>
  </form>
</div>
""")

@app.post("/easy")
async def easy_run(
    file: UploadFile = File(...),
    segment_seconds: int = Form(20),
    fast: int = Form(1),
    limit_segments: int = Form(3),
    preset: str = Form("fast"),
    workers: int = Form(2),
    simulate: str = Form("true"),
):
    """Easy runner endpoint"""
    if not RUNPOD_ENDPOINT:
        raise HTTPException(500, "RUNPOD_ENDPOINT not set")

    # 1) Save upload to temp
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_in = tempfile.mktemp(suffix=suffix)
    with open(tmp_in, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    # 2) Segment locally
    job_up = f"up_{uuid.uuid4().hex[:8]}"
    seg_paths = segment_copy_only_local(tmp_in, seg_seconds=segment_seconds, job_id=job_up)
    
    # For now, just use local paths
    signed_urls = [str(p) for p in seg_paths]
    signed_urls = signed_urls[: max(1, min(limit_segments, len(signed_urls)))]

    # cleanup local temps
    try:
        Path(tmp_in).unlink(missing_ok=True)
    except Exception:
        pass

    # 4) Launch RunPod (with simulate toggle support)
    job_id = str(uuid.uuid4())[:8]
    payload = {
        "job_id": job_id,
        "segment_urls": signed_urls,
        "callback_url": f"{PUBLIC_BASE_URL}/progress/{job_id}",
        "preset": preset or "fast",
        "workers": int(workers or 2),
        "make_overlay": True,
        "simulate": (str(simulate).lower() == "true"),
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {RUNPOD_API_KEY}"}
    r = requests.post(RUNPOD_ENDPOINT, headers=headers, json={"input": payload}, timeout=60)
    if r.status_code >= 300:
        raise HTTPException(500, f"RunPod launch failed: {r.text}")

    # 5) Redirect to monitor
    return HTMLResponse(f"""
<!doctype html><meta charset="utf-8"/><title>Launched {job_id}</title>
<body style="font-family:system-ui;background:#0b1320;color:#e9eef6">
  <div style="display:grid;place-items:center;min-height:100vh">
    <div style="background:#101a2d;padding:22px 28px;border-radius:14px">
      <h2 style="margin:0 0 10px">Launched job <code>{job_id}</code></h2>
      <p>Segments: {len(signed_urls)} | simulate: {str(simulate).lower()}</p>
      <p><a style="color:#66f0d5" href="/monitor/{job_id}">Open monitor</a></p>
      <script>setTimeout(()=>location.href="/monitor/{job_id}",600);</script>
    </div>
  </div>
</body>
""")

@app.websocket("/ws/{job_id}")
async def ws(job_id: str, websocket: WebSocket):
    await websocket.accept()
    channels.setdefault(job_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        try:
            channels[job_id].remove(websocket)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

