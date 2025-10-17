# server/server.py
import os, uuid, json, tempfile, subprocess, time, logging
from pathlib import Path
from typing import Dict, List

import requests
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi import Form


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

# Supabase (server-side only, required by /upload in Step 2+)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "scoutiqo")

# Shared secret to protect the progress callback
CALLBACK_SECRET = os.getenv("CALLBACK_SECRET", "")

app = FastAPI(title="ScoutIQO Analyzer", docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("uvicorn")

# -------- static media root (kept for compatibility; /upload no longer uses this) --------
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

def transcode_and_segment_local(local_input: str, seg_seconds: int, job_id: str) -> list[Path]:
    """
    Normalize video then segment into a temporary directory.
    Returns a list of local Path objects to seg_XXX.mp4
    """
    tmp_norm = tempfile.mktemp(suffix=".mp4")
    _run_ffmpeg([
        "ffmpeg","-y","-hwaccel","auto","-i", local_input,
        "-vf","scale='min(1280,iw)':-2,fps=25",
        "-c:v","h264","-preset","veryfast","-crf","20",
        "-g","50","-keyint_min","50","-sc_threshold","0",
        "-c:a","aac","-ac","1","-ar","32000","-b:a","64k",
        tmp_norm
    ])
    out_dir = Path(tempfile.mkdtemp()) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        "ffmpeg","-y","-i", tmp_norm, "-reset_timestamps","1","-map","0","-c","copy",
        "-segment_time", str(seg_seconds), "-f","segment", str(out_dir / "seg_%03d.mp4")
    ])
    return sorted(out_dir.glob("seg_*.mp4"))

def segment_copy_only_local(local_input: str, seg_seconds: int, job_id: str) -> list[Path]:
    """
    Copy-only segmentation into a temporary directory.
    Returns a list of local Path objects to seg_XXX.mp4
    """
    out_dir = Path(tempfile.mkdtemp()) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        "ffmpeg","-y","-i", local_input, "-reset_timestamps","1","-map","0","-c","copy",
        "-segment_time", str(seg_seconds), "-f","segment", str(out_dir / "seg_%03d.mp4")
    ])
    return sorted(out_dir.glob("seg_*.mp4"))

# -------- Supabase storage client --------
def _guess_ct(path: str) -> str:
    import mimetypes
    return mimetypes.guess_type(path)[0] or "application/octet-stream"

def _sb_headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }

def _sb_base() -> str:
    return f"{SUPABASE_URL}/storage/v1"

def supabase_put_file(local_path: str, object_key: str) -> dict:
    """
    Upload local file to Supabase bucket at object_key.
    Returns: {"bucket": str, "key": str, "signed_url": str, "public_url": str}
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(500, "Supabase env not set (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)")
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(local_path)
    url = f"{_sb_base()}/object/{SUPABASE_BUCKET}/{object_key}"
    with open(p, "rb") as f:
        r = requests.post(url, headers={**_sb_headers(), "Content-Type": _guess_ct(str(p))}, data=f)
    if r.status_code >= 300:
        raise HTTPException(500, f"Supabase upload failed {r.status_code}: {r.text[:400]}")
    # signed url (24h)
    signed = supabase_sign_url(object_key, expires_in=24*3600)
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{object_key}"
    return {"bucket": SUPABASE_BUCKET, "key": object_key, "signed_url": signed, "public_url": public_url}

def supabase_sign_url(object_key: str, expires_in: int = 3600) -> str:
    url = f"{_sb_base()}/object/sign/{SUPABASE_BUCKET}/{object_key}"
    r = requests.post(url, headers=_sb_headers(), json={"expiresIn": expires_in})
    if r.status_code >= 300:
        raise HTTPException(500, f"Supabase sign failed {r.status_code}: {r.text[:400]}")
    signed_path = (r.json().get("signedURL") or r.json().get("signedUrl"))
    if not signed_path:
        raise HTTPException(500, f"Supabase sign response missing signedURL: {r.text[:400]}")
    return f"{SUPABASE_URL}/storage/v1{signed_path}"

# -------- API --------
@app.get("/")
def root():
    return {"ok": True, "health": "/health", "docs": "/docs", "monitor": "/monitor/<job_id>"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile = File(...), segment_seconds: int = 20, fast: int = 0):
    """
    Step 2 version:
      1) save upload to a temp file
      2) segment locally into temp dir
      3) upload each seg to Supabase Storage
      4) return SIGNED URLs (not /media)
    """
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

    # Segment locally (temp dir)
    seg_paths = (
        segment_copy_only_local(tmp_in, seg_seconds=segment_seconds, job_id=job_id)
        if fast else
        transcode_and_segment_local(tmp_in, seg_seconds=segment_seconds, job_id=job_id)
    )

    # Upload each segment to Supabase
    signed_urls: List[str] = []
    for p in seg_paths:
        object_key = f"jobs/{job_id}/{p.name}"
        info = supabase_put_file(str(p), object_key)
        signed_urls.append(info["signed_url"])

    # Cleanup temp files (best-effort)
    try:
        Path(tmp_in).unlink(missing_ok=True)
        for p in seg_paths:
            p.unlink(missing_ok=True)
        # remove empty parent directory if any
        if seg_paths:
            seg_paths[0].parent.rmdir()
    except Exception:
        pass

    return {"job_id": job_id, "segments": signed_urls, "segment_urls": signed_urls}

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

# --- progress capture (protected by shared secret) ---
@app.post("/progress/{job_id}")
async def progress(job_id: str, req: Request, x_callback_token: str | None = Header(None)):
    # Require secret header from RunPod
    if not CALLBACK_SECRET or x_callback_token != CALLBACK_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

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
<!doctype html><meta charset="utf-8"/><title>Monitor â€“ {job_id}</title>
<style>
  body{{font-family:system-ui;margin:20px}}
  .row{{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}}
  video{{width:640px;height:360px;background:#000;border-radius:8px}}
  pre{{max-height:520px;overflow:auto;background:#111;color:#eee;padding:12px;border-radius:8px;min-width:420px}}
  .pill{{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef;margin-left:8px}}
</style>
<h1>Job <code>{job_id}</code> <span id="badge" class="pill">waitingâ€¦</span></h1>
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
    badge.textContent = st.last_type ? st.last_type : "waitingâ€¦";

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
@app.get("/easy")
def easy():
    # Simple HTML page that runs the full pipeline from the browser
    return HTMLResponse(f"""
<!doctype html><meta charset="utf-8"/><title>ScoutIQO â€“ Easy Runner</title>
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
  <h1>ScoutIQO â€“ Easy Runner <span class="ok">BASE: {PUBLIC_BASE_URL}</span></h1>
  <form action="/easy" method="post" enctype="multipart/form-data">
    <label>Video file (.mp4)</label>
    <input type="file" name="file" accept="video/mp4" required>

    <div class="row">
      <div>
        <label>Segment seconds</label>
        <input type="number" name="segment_seconds" value="12" min="4" max="60" />
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
    <div class="hint">This page: uploads â†’ segments â†’ uploads to Supabase â†’ launches RunPod â†’ redirects to monitor.</div>
  </form>
</div>
""")

@app.post("/easy")
async def easy_run(
    file: UploadFile = File(...),
    segment_seconds: int = Form(12),
    fast: int = Form(1),
    limit_segments: int = Form(3),
    preset: str = Form("fast"),
    workers: int = Form(2),
    simulate: str = Form("true"),
):
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
    seg_paths = (
        segment_copy_only_local(tmp_in, seg_seconds=segment_seconds, job_id=job_up)
        if fast else
        transcode_and_segment_local(tmp_in, seg_seconds=segment_seconds, job_id=job_up)
    )

    # 3) Upload segments to Supabase
    signed_urls: List[str] = []
    for p in seg_paths:
        object_key = f"jobs/{job_up}/{p.name}"
        info = supabase_put_file(str(p), object_key)
        signed_urls.append(info["signed_url"])

    # limit segments for quick test
    signed_urls = signed_urls[: max(1, min(limit_segments, len(signed_urls)))]

    # cleanup local temps
    try:
        Path(tmp_in).unlink(missing_ok=True)
        for p in seg_paths:
            p.unlink(missing_ok=True)
        if seg_paths:
            seg_paths[0].parent.rmdir()
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

