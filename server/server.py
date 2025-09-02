# server.py
import os, uuid, json, tempfile, subprocess, time, logging
from pathlib import Path
from typing import Dict, List

import requests
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- .env support (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------- Config --------
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "").strip()  # e.g. https://api.runpod.ai/v2/<endpoint-id>/run
RUNPOD_API_KEY  = os.getenv("RUNPOD_API_KEY", "").strip()

app = FastAPI(title="ScoutIQO Analyzer", docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("uvicorn")

# -------- static media root (segments served at /media/<job_id>/seg_XXX.mp4) --------
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

# -------- in-memory progress store --------
progress_store: Dict[str, List[dict]] = {}

# -------- ffmpeg helpers --------
def _run_ffmpeg(cmd: list):
    log.info("FFmpeg: %s", " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        out = p.stdout.decode(errors="ignore")
        log.error("FFmpeg failed:\n%s", out[:2000])
        raise RuntimeError(out)

def transcode_and_segment(local_input: str, seg_seconds: int, job_id: str) -> list[str]:
    # normalize -> 1280p max, 25fps, GOP=2s -> then segment
    tmp_norm = tempfile.mktemp(suffix=".mp4")
    _run_ffmpeg([
        "ffmpeg","-y","-hwaccel","auto","-i", local_input,
        "-vf","scale='min(1280,iw)':-2,fps=25",
        "-c:v","h264","-preset","veryfast","-crf","20",
        "-g","50","-keyint_min","50","-sc_threshold","0",
        "-c:a","aac","-ac","1","-ar","32000","-b:a","64k",
        tmp_norm
    ])
    out_dir = MEDIA_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        "ffmpeg","-y","-i", tmp_norm, "-reset_timestamps","1","-map","0","-c","copy",
        "-segment_time", str(seg_seconds), "-f","segment", str(out_dir / "seg_%03d.mp4")
    ])
    return [f"{PUBLIC_BASE_URL}/media/{job_id}/{p.name}" for p in sorted(out_dir.glob("seg_*.mp4"))]

def segment_copy_only(local_input: str, seg_seconds: int, job_id: str) -> list[str]:
    out_dir = MEDIA_ROOT / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        "ffmpeg","-y","-i", local_input, "-reset_timestamps","1","-map","0","-c","copy",
        "-segment_time", str(seg_seconds), "-f","segment", str(out_dir / "seg_%03d.mp4")
    ])
    return [f"{PUBLIC_BASE_URL}/media/{job_id}/{p.name}" for p in sorted(out_dir.glob("seg_*.mp4"))]

# -------- API --------
@app.get("/")
def root():
    return {"ok": True, "health": "/health", "docs": "/docs", "monitor": "/monitor/<job_id>"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile = File(...), segment_seconds: int = 20, fast: int = 1):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_in = tempfile.mktemp(suffix=suffix)
    job_id = f"up_{uuid.uuid4().hex[:8]}"
    with open(tmp_in, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    segs = (
        segment_copy_only(tmp_in, seg_seconds=segment_seconds, job_id=job_id)
        if fast else
        transcode_and_segment(tmp_in, seg_seconds=segment_seconds, job_id=job_id)
    )
    return {"job_id": job_id, "segments": segs}

class AnalyzeReq(BaseModel):
    segment_urls: List[str]
    preset: str | None = "fast"
    workers: int | None = 4

@app.post("/analyze")
def analyze(req: AnalyzeReq):
    if not RUNPOD_ENDPOINT:
        raise HTTPException(500, "RUNPOD_ENDPOINT not set")
    job_id = str(uuid.uuid4())[:8]
    if not req.segment_urls:
        raise HTTPException(400, "segment_urls required")
    payload = {
        "job_id": job_id,
        "segment_urls": req.segment_urls,
        "callback_url": f"{PUBLIC_BASE_URL}/progress/{job_id}",
        "preset": req.preset or "fast",
        "workers": int(req.workers or 4),
        "make_overlay": True,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {RUNPOD_API_KEY}"}
    r = requests.post(RUNPOD_ENDPOINT, headers=headers, json={"input": payload}, timeout=60)
    if r.status_code >= 300:
        raise HTTPException(500, f"RunPod launch failed: {r.text}")
    return {"job_id": job_id}

# --- progress capture ---
@app.post("/progress/{job_id}")
async def progress(job_id: str, req: Request):
    data = await req.json()
    progress_store.setdefault(job_id, []).append({"ts": time.time(), **data})
    logging.info("PROGRESS %s: %s", job_id, data)
    await publish(job_id, data)
    return {"ok": True}

@app.get("/status/{job_id}")
def status(job_id: str):
    evts = progress_store.get(job_id, [])
    return {"job_id": job_id, "events": len(evts), "last_type": (evts[-1]["type"] if evts else None)}

@app.get("/progress/{job_id}/dump")
def dump(job_id: str):
    return {"job_id": job_id, "events": progress_store.get(job_id, [])}

# --- minimal monitor page ---
@app.get("/monitor/{job_id}")
def monitor(job_id: str):
    return HTMLResponse(f"""
<!doctype html><meta charset="utf-8"/><title>Monitor – {job_id}</title>
<style>
  body{{font-family:system-ui;margin:20px}}
  .row{{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}}
  video{{width:640px;height:360px;background:#000;border-radius:8px}}
  pre{{max-height:520px;overflow:auto;background:#111;color:#eee;padding:12px;border-radius:8px;min-width:420px}}
  .pill{{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef;margin-left:8px}}
</style>
<h1>Job <code>{job_id}</code> <span id="badge" class="pill">waiting…</span></h1>
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
          try {{ await vid.play(); }} catch (e) {{ /* autoplay may be blocked; user can click play */ }}
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
