

# --- monitor2 overlay viewer (safe, no decorators) ---
def _monitor2_endpoint(job_id: str):
    import json
    from fastapi.responses import HTMLResponse
    job_js = json.dumps(job_id)
    html = r"""<!doctype html><meta charset="utf-8"/><title>Monitor</title>
<style>
  body{font-family:system-ui;margin:20px}
  .row{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}
  video{width:640px;height:360px;background:#000;border-radius:8px}
  pre{max-height:520px;overflow:auto;background:#111;color:#eee;padding:12px;border-radius:8px;min-width:420px}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef;margin-left:8px}
  ul#arts{margin:8px 0 0 0;padding:0;list-style:none}
  ul#arts li{margin:4px 0}
  ul#arts a{text-decoration:none}
</style>
<h1>Job <code>__JOB_TEXT__</code> <span id="badge" class="pill">waiting…</span></h1>
<div class="row">
  <div>
    <video id="vid" controls muted playsinline></video>
    <div style="margin-top:8px">
      <button onclick="forcePoll()">Refresh now</button>
      <a id="dump" target="_blank">Open full event dump</a>
    </div>
    <div style="margin-top:12px">
      <h3 style="margin:0 0 6px">Artifacts</h3>
      <ul id="arts"></ul>
    </div>
  </div>
  <pre id="log"></pre>
</div>
<script>
const job   = __JOB_JSON__;
const base  = location.origin;
const logEl = document.getElementById('log');
const vid   = document.getElementById('vid');
const badge = document.getElementById('badge');
const dumpA = document.getElementById('dump');
const artsUl= document.getElementById('arts');

let lastCount = 0;
let lastVideoSrc = "";

// prefer overlay videos if present
function pickBestVideoArtifact(arts) {
  if (vids.length === 0) return null;
  vids.sort((a,b) => (b.name && b.name.includes('overlay') ? 1 : 0) - (a.name && a.name.includes('overlay') ? 1 : 0));
  return vids.pop();
}

async function poll() {
  try {
    badge.textContent = st.last_type ? st.last_type : "waiting…";

    if ((st.events || 0) !== lastCount) {
      lastCount = st.events || 0;

      logEl.textContent = JSON.stringify(dump, null, 2);
      const evts = (dump && dump.events) ? dump.events : [];

      const arts = [];
      for (const e of evts) {
        if (e.type === "artifact" && e.url) {
          arts.push({ name: e.name || "artifact", url: e.url, seg: e.seg });
        }
      }
      artsUl.innerHTML = "";
      for (const a of arts) {
        const li = document.createElement('li');
        const segText = (a.seg != null ? ('seg ' + a.seg + ': ') : '');
        artsUl.appendChild(li);
      }

      const best = pickBestVideoArtifact(arts);
      if (best && best.url !== lastVideoSrc) {
        lastVideoSrc = best.url;
        vid.src = best.url;
        try { await vid.play(); } catch (e) {}
      } else {
        const latest = evts.length ? evts[evts.length - 1] : null;
        if (latest && latest.type === "segment_start" && latest.url && latest.url !== lastVideoSrc) {
          lastVideoSrc = latest.url;
          vid.src = latest.url;
          try { await vid.play(); } catch (e) {}
        }
      }
    }
  } catch (e) { badge.textContent = "error"; }
}
function forcePoll(){ lastCount = -1; poll(); }
setInterval(poll, 1000); poll();
</script>
"""
    html = html.replace("__JOB_JSON__", job_js).replace("__JOB_TEXT__", job_id)
    return HTMLResponse(html)

# Register route AFTER app exists
try:
    app.add_api_route("/monitor2/{job_id}", _monitor2_endpoint, methods=["GET"])
except Exception:
    # if app not yet defined at import time, a later import can re-run this or you can move these lines lower
    pass
# --- end monitor2 ---

# --- ensure FastAPI app exists (auto-added) ---
try:
    app  # noqa: F821
except NameError:
    from fastapi import FastAPI
    app = FastAPI()

# minimal health endpoint so we can verify the server runs
@app.get("/health")
def _health():
    return {"ok": True}
# --- end ensure ---


@app.get("/_health")
def _health():
    import os
    env = {k: bool(os.getenv(k)) for k in [
        "SUPABASE_URL","SUPABASE_SERVICE_ROLE_KEY","SUPABASE_BUCKET",
        "PUBLIC_BASE_URL","CALLBACK_SECRET","FFMPEG_BIN"
    ]}
    return {"ok": True, "env": env}


# --- LOCAL UPLOAD + STATIC FILES (minimal) ------------------------------------
try:
    import os, uuid, shutil, subprocess
    from pathlib import Path
    from fastapi import UploadFile, File
    from fastapi.responses import JSONResponse
    from starlette.staticfiles import StaticFiles

    # Serve tmp files at /files
    TMP_ROOT = (Path(__file__).parent.parent / "tmp_www")
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        app.mount("/files", StaticFiles(directory=str(TMP_ROOT)), name="files")
    except Exception:
        # already mounted or not available – ignore
        pass

    @app.post("/upload")
    async def upload(file: UploadFile = File(...), segment_seconds: int = 12, fast: int = 1):
        ffmpeg = os.getenv("FFMPEG_BIN", "ffmpeg")
        base  = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8081")

        job_id = f"up_{uuid.uuid4().hex[:8]}"
        job_dir = TMP_ROOT / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # save input
        src = job_dir / "input.mp4"
        with src.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # segment -> seg_000.mp4, seg_001.mp4, ...
        pattern = job_dir / "seg_%03d.mp4"
        cmd = [
            ffmpeg, "-y",
            "-i", str(src),
            "-reset_timestamps", "1",
            "-map", "0", "-c", "copy",
            "-segment_time", str(segment_seconds),
            "-f", "segment", str(pattern)
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return JSONResponse({"detail":"ffmpeg failed", "stderr":proc.stderr[-6000:]}, status_code=500)

        segs = sorted(p for p in job_dir.glob("seg_*.mp4"))
        urls = [f"{base}/files/jobs/{job_id}/{p.name}" for p in segs]

        return {"job_id": job_id, "segment_urls": urls}
except Exception as _e:
    # keep import from crashing if anything above fails
    pass
# -------------------------------------------------------------------------------


# --- LOCAL UPLOAD + STATIC FILES (minimal) ------------------------------------
try:
    import os, uuid, shutil, subprocess
    from pathlib import Path
    from fastapi import UploadFile, File
    from fastapi.responses import JSONResponse
    from starlette.staticfiles import StaticFiles

    # Serve tmp files at /files
    TMP_ROOT = (Path(__file__).parent.parent / "tmp_www")
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        app.mount("/files", StaticFiles(directory=str(TMP_ROOT)), name="files")
    except Exception:
        # already mounted or not available – ignore
        pass

    @app.post("/upload")
    async def upload(file: UploadFile = File(...), segment_seconds: int = 12, fast: int = 1):
        ffmpeg = os.getenv("FFMPEG_BIN", "ffmpeg")
        base  = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8081")

        job_id = f"up_{uuid.uuid4().hex[:8]}"
        job_dir = TMP_ROOT / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # save input
        src = job_dir / "input.mp4"
        with src.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # segment -> seg_000.mp4, seg_001.mp4, ...
        pattern = job_dir / "seg_%03d.mp4"
        cmd = [
            ffmpeg, "-y",
            "-i", str(src),
            "-reset_timestamps", "1",
            "-map", "0", "-c", "copy",
            "-segment_time", str(segment_seconds),
            "-f", "segment", str(pattern)
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return JSONResponse({"detail":"ffmpeg failed", "stderr":proc.stderr[-6000:]}, status_code=500)

        segs = sorted(p for p in job_dir.glob("seg_*.mp4"))
        urls = [f"{base}/files/jobs/{job_id}/{p.name}" for p in segs]

        return {"job_id": job_id, "segment_urls": urls}
except Exception as _e:
    # keep import from crashing if anything above fails
    pass
# -------------------------------------------------------------------------------


# --- SIMPLE MONITOR PAGE -------------------------------------------------------
try:
    from fastapi.responses import HTMLResponse
    import json as _json

    @app.get("/monitor/{job_id}")
    def monitor(job_id: str):
        job_js = _json.dumps(job_id)
        html = r"""<!doctype html><meta charset="utf-8"/><title>Monitor</title>
<style>
  body{font-family:system-ui;margin:20px}
  .row{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}
  video{width:640px;height:360px;background:#000;border-radius:8px}
  pre{max-height:520px;overflow:auto;background:#111;color:#eee;padding:12px;border-radius:8px;min-width:420px}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef;margin-left:8px}
  ul#arts{margin:8px 0 0 0;padding:0;list-style:none}
  ul#arts li{margin:4px 0}
  ul#arts a{text-decoration:none}
</style>
<h1>Job <code>__JOB_TEXT__</code> <span id="badge" class="pill">waiting…</span></h1>
<div class="row">
  <div>
    <video id="vid" controls muted playsinline></video>
    <div style="margin-top:8px">
      <button onclick="forcePoll()">Refresh now</button>
      <a id="dump" target="_blank">Open full event dump</a>
    </div>
    <div style="margin-top:12px">
      <h3 style="margin:0 0 6px">Artifacts</h3>
      <ul id="arts"></ul>
    </div>
  </div>
  <pre id="log"></pre>
</div>
<script>
const job   = __JOB_JSON__;
const base  = location.origin;
const logEl = document.getElementById('log');
const vid   = document.getElementById('vid');
const badge = document.getElementById('badge');
const dumpA = document.getElementById('dump');
const artsUl= document.getElementById('arts');

let lastCount = -1;
let lastVideoSrc = "";

function pickBestVideoArtifact(arts) {
  if (vids.length === 0) return null;
  vids.sort((a,b) => (b.name && b.name.includes('overlay') ? 1 : 0) - (a.name && a.name.includes('overlay') ? 1 : 0));
  return vids.pop();
}

async function poll() {
  try {
    badge.textContent = st.last_type || "waiting…";

    // Only refresh UI when event count changes
    if ((st.events || 0) !== lastCount) {
      lastCount = st.events || 0;

      logEl.textContent = JSON.stringify(dump, null, 2);
      const evts = (dump && dump.events) ? dump.events : [];

      // Build artifacts list
      const arts = [];
      for (const e of evts) {
        if (e.type === "artifact" && e.url) {
          arts.push({ name: e.name || "artifact", url: e.url, seg: e.seg });
        }
      }
      artsUl.innerHTML = "";
      for (const a of arts) {
        const li = document.createElement('li');
        const segText = (a.seg != null ? ('seg ' + a.seg + ': ') : '');
        artsUl.appendChild(li);
      }

      // Prefer overlay artifact; else show the last segment_start URL
      const best = pickBestVideoArtifact(arts);
      if (best && best.url !== lastVideoSrc) {
        lastVideoSrc = best.url;
        vid.src = best.url;
        try { await vid.play(); } catch (e) {}
      } else {
        const latest = evts.length ? evts[evts.length - 1] : null;
        if (latest && latest.type === "segment_start" && latest.url && latest.url !== lastVideoSrc) {
          lastVideoSrc = latest.url;
          vid.src = latest.url;
          try { await vid.play(); } catch (e) {}
        }
      }
    }
  } catch (e) {
    badge.textContent = "error";
  }
}

function forcePoll(){ lastCount = -1; poll(); }
setInterval(poll, 1000); poll();
</script>
"""
        html = html.replace("__JOB_JSON__", job_js).replace("__JOB_TEXT__", job_id)
        return HTMLResponse(html)
except Exception:
    pass
# -------------------------------------------------------------------------------


# ---- PROGRESS & STATUS (minimal, in-memory) -----------------------------------
import os
from fastapi import Request, HTTPException

if "JOBS" not in globals():
    JOBS = {}  # job_id -> {"events":[...], "last_type":None}

def _job(jid:str):
    if jid not in JOBS:
        JOBS[jid] = {"events": [], "last_type": None}
    return JOBS[jid]

@app.post("/progress/{job_id}")
async def progress(job_id: str, req: Request):
    # Optional: enforce secret
    want = os.environ.get("CALLBACK_SECRET") or ""
    got  = req.headers.get("x-callback-token", "")
    if want and got != want:
        raise HTTPException(status_code=401, detail="bad token")

    payload = await req.json()
    j = _job(job_id)

    # store event and update last_type
    et = payload.get("type")
    j["events"].append(payload)
    j["last_type"] = et

    # You could persist to Supabase here; for demo we just keep it in memory
    return {"ok": True}

@app.get("/status/{job_id}")
def status(job_id: str):
    j = _job(job_id)
    return {"ok": True, "events": len(j["events"]), "last_type": j["last_type"]}

@app.get("/progress/{job_id}/dump")
def dump(job_id: str):
    j = _job(job_id)
    return {"ok": True, "events": j["events"]}
# ------------------------------------------------------------------------------


# ---- analyze_local endpoint (background OpenCV overlay) ----------------------
try:
    from pydantic import BaseModel
    from fastapi import BackgroundTasks
    import os, requests, json
    from overlay_analyzer import make_overlay_for_segment

    class AnalyzeReq(BaseModel):
        job_id: str
        segment_urls: list[str]

    def _post_progress(job_id: str, payload: dict):
        base  = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8081")
        token = os.environ.get("CALLBACK_SECRET", "scoutsecret123")
        requests.post(f"{base}/progress/{job_id}", json=payload,
                      headers={"x-callback-token": token}, timeout=30).raise_for_status()

    def _run_local_overlay(job_id: str, segs: list[str]):
        for i, url in enumerate(segs):
            _post_progress(job_id, {"type":"segment_start","seg":i,"url":url})
            _post_progress(job_id, {"type":"status","seg":i,"msg":"analyzing (opencv)..."})
            overlay_rel = make_overlay_for_segment(url, job_id, i)      # "/files/jobs/<job>/overlay_seg_000.mp4"
            overlay_url = f"{os.environ.get('PUBLIC_BASE_URL','http://127.0.0.1:8081')}{overlay_rel}"
            _post_progress(job_id, {"type":"artifact","seg":i,"name":f"overlay_seg_{i:03}.mp4","url":overlay_url})
            _post_progress(job_id, {"type":"segment_done","seg":i})
        _post_progress(job_id, {"type":"job_done"})

    @app.post("/analyze_local")
    def analyze_local(req: AnalyzeReq, bg: BackgroundTasks):
        # fire-and-forget background job
        bg.add_task(_run_local_overlay, req.job_id, req.segment_urls)
        return {"ok": True, "job_id": req.job_id, "segments": len(req.segment_urls)}
except Exception as _e:
    # keep server booting even if overlay code missing
    pass
# -----------------------------------------------------------------------------


# =================== LIVE MJPEG STREAM ===================
try:
    from fastapi.responses import StreamingResponse
    import cv2, time
    from .live_core import (
        init_tracker_state, run_player_detector, run_tracker,
        detect_ball, assign_teams, SpeedSmoother,
        compute_control, draw_overlay
    )

    @app.get("/live")
    def live(src: str, resize_w: int = 1280, skip: int = 1):
        """
        Open in browser:
        /live?src=C:\\\\path\\\\to\\\\test.mp4
        or a public/local segment URL.
        """
        def gen():
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {src}")
            st = init_tracker_state()
            smooth = SpeedSmoother()
            i = 0
            
            
            last_ball = None
            while True:
                ok, frame = cap.read()
                if not ok: break
                if resize_w:
                    h, w = frame.shape[:2]
                    frame = cv2.resize(frame, (resize_w, int(resize_w*h/w)))
                if i % max(1, skip) == 0:
                    dets = run_player_detector(frame)
                    tids, tracks = run_tracker(st, dets)
                    ball = detect_ball(frame)
                    assign_teams(st, frame, tracks)
                    speeds = smooth.update(st, tracks)
                    control = compute_control(st, tracks, ball)
                    draw_overlay(frame, tracks, speeds, ball, st.team_of, control)
                i += 1
                ok, jpg = cv2.imencode(".jpg", frame)
                if not ok: continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            cap.release()
            # persist tuned values
            try:
                save_state(
                    m_per_px=float(st.m_per_px),
                    exclude_top_pct=float(exclude_top_pct),
                    min_overlap=float(min_overlap),
                    conf_min=float(conf_min)
                )
            except Exception:
                pass

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
except Exception as _e:
    # keep API alive even if OpenCV/live_core not present
    pass
# =========================================================


# =================== LIVE2 (pitch-filtered MJPEG stream with metrics) ===================
try:
    from fastapi.responses import StreamingResponse
    import cv2, time
    from .live_core import (
        init_tracker_state, run_player_detector, run_tracker,
        detect_ball, assign_teams, SpeedEstimator, estimate_m_per_px,
        compute_control, draw_overlay, estimate_pitch_bounds
    )
    from .pitch import build_pitch_mask, box_kept_by_mask
    from .numbering import SquadNumberer
    from .metrics_core import (
        MetricsState, nearest_player_to_ball, in_final_third,
        header_likely, tackle_likely, pass_likely
    )
    try:
        from .learn_state import load_state, save_state
    except Exception:
        def load_state(): return {"m_per_px":0.25,"exclude_top_pct":0.25,"min_overlap":0.60,"conf_min":0.60}
        def save_state(**kwargs): pass

    # global live metrics snapshot (very small)
    LIVE_METRICS = {"ok": False, "snapshot": {}}

    @app.get("/metrics")
    def metrics():
        return LIVE_METRICS

    @app.get("/live2")
    def live2(
        src: str,
        resize_w: int = 1280,
        skip: int = 1,
        exclude_top_pct: float | None = None,
        min_overlap: float | None = None,
        conf_min: float | None = None
    ):
        _s = load_state()
        if exclude_top_pct is None: exclude_top_pct = float(_s.get("exclude_top_pct", 0.25))
        if min_overlap is None:     min_overlap     = float(_s.get("min_overlap", 0.60))
        if conf_min is None:        conf_min        = float(_s.get("conf_min", 0.60))

        def gen():
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {src}")

            st = init_tracker_state()
            metr = MetricsState()
            numr = SquadNumberer()

            # first frame: mask/scale/bounds
            ok, frame = cap.read()
            if not ok:
                cap.release(); raise RuntimeError("No frames.")
            if resize_w:
                h0, w0 = frame.shape[:2]
                frame = cv2.resize(frame, (resize_w, int(resize_w*h0/w0)))

            H, W = frame.shape[:2]
            excl = (0, 0, W, int(H*exclude_top_pct))
            st.pitch_mask = build_pitch_mask(frame, exclude_rect=excl)
            bounds = estimate_pitch_bounds(st.pitch_mask)
            st.m_per_px = estimate_m_per_px(st.pitch_mask)
            speed = SpeedEstimator(m_per_px=st.m_per_px)

            i = 0
            while True:
                if i>0:
                    ok, frame = cap.read()
                    if not ok: break
                    if resize_w:
                        h,w=frame.shape[:2]
                        frame = cv2.resize(frame, (resize_w, int(resize_w*h/w)))

                if i % max(1,skip)==0:
                    dets = run_player_detector(frame, conf=conf_min)
                    dets = [d for d in dets if box_kept_by_mask(d[:4], st.pitch_mask, min_overlap=min_overlap)]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    tids, tracks = run_tracker(st, dets, fps=fps)
                    assign_teams(st, frame, tracks)

                    # squad numbers & labels
                    for tid, box in tracks.items():
                        team = st.team_of.get(tid,0)
                        numr.touch(team, tid)
                    numr.gc()

                    # ball
                    raw_ball = detect_ball(frame)
                    ball_xy = None
                    if raw_ball:
                        x1,y1,x2,y2 = raw_ball
                        ball_xy = ((x1+x2)/2, (y1+y2)/2)
                    bx, by = metr.ball_trk.step(ball_xy, fps=fps)

                    # owner = nearest player under threshold
                    owner_tid, dist = nearest_player_to_ball((bx,by), tracks)
                    owner_team = st.team_of.get(owner_tid, 0) if owner_tid is not None else None
                    if owner_team is not None:
                        metr.update_possession(owner_team)

                        sq = numr.get(owner_team, owner_tid)
                        inside, xnorm = in_final_third((bx,by), bounds)
                        metr.mark_touch(owner_team, sq, in_final_third=inside)

                        # headers
                        if owner_tid in tracks and header_likely((bx,by), tracks[owner_tid]):
                            metr.mark_header(owner_team, sq)

                        # simple pass / tackle inference
                        prev = metr.current_owner
                        new  = (owner_team, sq)
                        if pass_likely(prev, new):
                            metr.mark_pass(prev[0], prev[1] or 0, new[0], new[1] or 0, success=True)
                        if tackle_likely(prev, new, dist):
                            metr.mark_tackle(new[0], new[1])

                    # speeds and control + overlay with A/B# numbers
                    speeds = speed.update(st, tracks)
                    control = compute_control(st, tracks, [bx-5,by-5,bx+5,by+5])

                    # replace draw_overlay labels with numbered ones
                    for tid, box in tracks.items():
                        team = st.team_of.get(tid,0)
                        sq = numr.get(team, tid)
                        # draw number near feet
                        x1,y1,x2,y2 = map(int, box)
                        color = (0,255,0) if team==0 else (255,255,0)
                        tag = ("A" if team==0 else "B") + str(sq)
                        cv2.putText(frame, f"{tag} {speeds.get(tid,0.0):.1f} km/h",
                                    (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                        cx=int((x1+x2)/2); cy=int(y2); cv2.circle(frame,(cx,cy),8,color,2)

                    # draw ball
                    cv2.circle(frame, (int(bx),int(by)), 6, (0,0,255), -1)

                    # possession panel
                    tot = max(1e-3, sum(metr.possession))
                    pA = 100*metr.possession[0]/tot; pB = 100*metr.possession[1]/tot
                    panel = (np.zeros((60,260,3), dtype=np.uint8))
                    cv2.putText(panel, f"Team A Control: {pA:5.1f}%", (8,22), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,255,200),1,cv2.LINE_AA)
                    cv2.putText(panel, f"Team B Control: {pB:5.1f}%", (8,48), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,255,200),1,cv2.LINE_AA)
                    h,w=frame.shape[:2]; frame[h-60:h, w-260:w] = cv2.addWeighted(frame[h-60:h, w-260:w],0.3,panel,0.7,0)

                    # formation sketch input (store centers per team)
                    A=[]; B=[]
                    for tid, b in tracks.items():
                        cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2
                        (A if st.team_of.get(tid,0)==0 else B).append((cx,cy))
                    metr.form_hist.append((np.array(A), np.array(B)))

                    # publish snapshot
                    LIVE_METRICS["ok"]=True
                    LIVE_METRICS["snapshot"]=metr.snapshot()

                i+=1
                ok, jpg = cv2.imencode(".jpg", frame)
                if not ok: continue
                yield (b"--frame\\r\\nContent-Type: image/jpeg\\r\\n\\r\\n" + jpg.tobytes() + b"\\r\\n")

            cap.release()
            try:
                save_state(
                    m_per_px=float(st.m_per_px),
                    exclude_top_pct=float(exclude_top_pct),
                    min_overlap=float(min_overlap),
                    conf_min=float(conf_min)
                )
            except Exception:
                pass

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
except Exception:
    pass
# ==========================================================================


# =================== LIVE2 (pitch-filtered MJPEG stream with metrics) ===================
try:
    from fastapi.responses import StreamingResponse
    import cv2, time
    from .live_core import (
        init_tracker_state, run_player_detector, run_tracker,
        detect_ball, assign_teams, SpeedEstimator, estimate_m_per_px,
        compute_control, draw_overlay, estimate_pitch_bounds
    )
    from .pitch import build_pitch_mask, box_kept_by_mask
    from .numbering import SquadNumberer
    from .metrics_core import (
        MetricsState, nearest_player_to_ball, in_final_third,
        header_likely, tackle_likely, pass_likely
    )
    try:
        from .learn_state import load_state, save_state
    except Exception:
        def load_state(): return {"m_per_px":0.25,"exclude_top_pct":0.25,"min_overlap":0.60,"conf_min":0.60}
        def save_state(**kwargs): pass

    # global live metrics snapshot (very small)
    LIVE_METRICS = {"ok": False, "snapshot": {}}

    @app.get("/metrics")
    def metrics():
        return LIVE_METRICS

    @app.get("/live2")
    def live2(
        src: str,
        resize_w: int = 1280,
        skip: int = 1,
        exclude_top_pct: float | None = None,
        min_overlap: float | None = None,
        conf_min: float | None = None
    ):
        _s = load_state()
        if exclude_top_pct is None: exclude_top_pct = float(_s.get("exclude_top_pct", 0.25))
        if min_overlap is None:     min_overlap     = float(_s.get("min_overlap", 0.60))
        if conf_min is None:        conf_min        = float(_s.get("conf_min", 0.60))

        def gen():
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {src}")

            st = init_tracker_state()
            metr = MetricsState()
            numr = SquadNumberer()

            # first frame: mask/scale/bounds
            ok, frame = cap.read()
            if not ok:
                cap.release(); raise RuntimeError("No frames.")
            if resize_w:
                h0, w0 = frame.shape[:2]
                frame = cv2.resize(frame, (resize_w, int(resize_w*h0/w0)))

            H, W = frame.shape[:2]
            excl = (0, 0, W, int(H*exclude_top_pct))
            st.pitch_mask = build_pitch_mask(frame, exclude_rect=excl)
            bounds = estimate_pitch_bounds(st.pitch_mask)
            st.m_per_px = estimate_m_per_px(st.pitch_mask)
            speed = SpeedEstimator(m_per_px=st.m_per_px)

            i = 0
            while True:
                if i>0:
                    ok, frame = cap.read()
                    if not ok: break
                    if resize_w:
                        h,w=frame.shape[:2]
                        frame = cv2.resize(frame, (resize_w, int(resize_w*h/w)))

                if i % max(1,skip)==0:
                    dets = run_player_detector(frame, conf=conf_min)
                    dets = [d for d in dets if box_kept_by_mask(d[:4], st.pitch_mask, min_overlap=min_overlap)]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    tids, tracks = run_tracker(st, dets, fps=fps)
                    assign_teams(st, frame, tracks)

                    # squad numbers & labels
                    for tid, box in tracks.items():
                        team = st.team_of.get(tid,0)
                        numr.touch(team, tid)
                    numr.gc()

                    # ball
                    raw_ball = detect_ball(frame)
                    ball_xy = None
                    if raw_ball:
                        x1,y1,x2,y2 = raw_ball
                        ball_xy = ((x1+x2)/2, (y1+y2)/2)
                    bx, by = metr.ball_trk.step(ball_xy, fps=fps)

                    # owner = nearest player under threshold
                    owner_tid, dist = nearest_player_to_ball((bx,by), tracks)
                    owner_team = st.team_of.get(owner_tid, 0) if owner_tid is not None else None
                    if owner_team is not None:
                        metr.update_possession(owner_team)

                        sq = numr.get(owner_team, owner_tid)
                        inside, xnorm = in_final_third((bx,by), bounds)
                        metr.mark_touch(owner_team, sq, in_final_third=inside)

                        # headers
                        if owner_tid in tracks and header_likely((bx,by), tracks[owner_tid]):
                            metr.mark_header(owner_team, sq)

                        # simple pass / tackle inference
                        prev = metr.current_owner
                        new  = (owner_team, sq)
                        if pass_likely(prev, new):
                            metr.mark_pass(prev[0], prev[1] or 0, new[0], new[1] or 0, success=True)
                        if tackle_likely(prev, new, dist):
                            metr.mark_tackle(new[0], new[1])

                    # speeds and control + overlay with A/B# numbers
                    speeds = speed.update(st, tracks)
                    control = compute_control(st, tracks, [bx-5,by-5,bx+5,by+5])

                    # replace draw_overlay labels with numbered ones
                    for tid, box in tracks.items():
                        team = st.team_of.get(tid,0)
                        sq = numr.get(team, tid)
                        # draw number near feet
                        x1,y1,x2,y2 = map(int, box)
                        color = (0,255,0) if team==0 else (255,255,0)
                        tag = ("A" if team==0 else "B") + str(sq)
                        cv2.putText(frame, f"{tag} {speeds.get(tid,0.0):.1f} km/h",
                                    (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                        cx=int((x1+x2)/2); cy=int(y2); cv2.circle(frame,(cx,cy),8,color,2)

                    # draw ball
                    cv2.circle(frame, (int(bx),int(by)), 6, (0,0,255), -1)

                    # possession panel
                    tot = max(1e-3, sum(metr.possession))
                    pA = 100*metr.possession[0]/tot; pB = 100*metr.possession[1]/tot
                    panel = (np.zeros((60,260,3), dtype=np.uint8))
                    cv2.putText(panel, f"Team A Control: {pA:5.1f}%", (8,22), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,255,200),1,cv2.LINE_AA)
                    cv2.putText(panel, f"Team B Control: {pB:5.1f}%", (8,48), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,255,200),1,cv2.LINE_AA)
                    h,w=frame.shape[:2]; frame[h-60:h, w-260:w] = cv2.addWeighted(frame[h-60:h, w-260:w],0.3,panel,0.7,0)

                    # formation sketch input (store centers per team)
                    A=[]; B=[]
                    for tid, b in tracks.items():
                        cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2
                        (A if st.team_of.get(tid,0)==0 else B).append((cx,cy))
                    metr.form_hist.append((np.array(A), np.array(B)))

                    # publish snapshot
                    LIVE_METRICS["ok"]=True
                    LIVE_METRICS["snapshot"]=metr.snapshot()

                i+=1
                ok, jpg = cv2.imencode(".jpg", frame)
                if not ok: continue
                yield (b"--frame\\r\\nContent-Type: image/jpeg\\r\\n\\r\\n" + jpg.tobytes() + b"\\r\\n")

            cap.release()
            try:
                save_state(
                    m_per_px=float(st.m_per_px),
                    exclude_top_pct=float(exclude_top_pct),
                    min_overlap=float(min_overlap),
                    conf_min=float(conf_min)
                )
            except Exception:
                pass

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
except Exception:
    pass
# ==========================================================================


try:
    from .learn_state import load_state, save_state
except Exception:
    load_state = lambda: {"m_per_px":0.25,"exclude_top_pct":0.25,"min_overlap":0.60,"conf_min":0.60}
    save_state = lambda **kw: None







# =================== LIVE_PRO (all-in-one stream + metrics) ===================
try:
    from fastapi.responses import StreamingResponse
    import cv2, numpy as np, time
    from .live_core import (
        init_tracker_state, run_player_detector, run_tracker,
        detect_ball, assign_teams, SpeedEstimator, estimate_m_per_px,
        compute_control, draw_overlay, estimate_pitch_bounds
    )
    from .pitch import build_pitch_mask, box_kept_by_mask
    from ultralytics import YOLO
    from .numbering import SquadNumberer
    from .metrics_core import MetricsState, nearest_to_ball, in_final_third

    try:
        from .learn_state import load_state, save_state
    except Exception:
        def load_state(): return {"m_per_px":0.25,"exclude_top_pct":0.25,"min_overlap":0.60,"conf_min":0.60}
        def save_state(**kwargs): pass

    LIVE_METRICS = {"ok": False, "snapshot": {}}

    @app.get("/metrics")
    def metrics():
        return LIVE_METRICS

    # YOLO object detector for BALL (COCO "sports ball") and PERSON
    _det_model = None
    def _ensure_det():
        global _det_model
        if _det_model is None:
            _det_model = YOLO("yolov8n.pt")  # auto-download

    def yolo_person_and_ball(frame, conf=0.35):
        _ensure_det()
        r = _det_model.predict(source=frame, conf=conf, verbose=False)
        boxes=[]; balls=[]
        names = _det_model.model.names
        for b in r[0].boxes:
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            c = int(b.cls[0].item())
            cf = float(b.conf[0].item())
            label = names.get(c, "")
            if c==0:   # person
                boxes.append([x1,y1,x2,y2,cf])
            elif label in ("sports ball","sportsball","ball"):  # safety across weights
                balls.append([x1,y1,x2,y2,cf])
        # best ball only
        ball = max(balls, key=lambda t:t[4]) if balls else None
        return boxes, ball

    @app.get("/live_pro")
    def live_pro(
        src: str,
        resize_w: int = 1280,
        skip: int = 1,
        exclude_top_pct: float | None = None,
        min_overlap: float | None = None,
        conf_min: float | None = None
    ):
        cfg = load_state()
        if exclude_top_pct is None: exclude_top_pct = float(cfg.get("exclude_top_pct", 0.25))
        if min_overlap is None:     min_overlap     = float(cfg.get("min_overlap", 0.60))
        if conf_min is None:        conf_min        = float(cfg.get("conf_min", 0.60))

        def gen():
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {src}")

            st  = init_tracker_state()
            metr= MetricsState()
            nums= SquadNumberer()

            # boot: first frame -> mask/bounds/scale
            ok, frame = cap.read()
            if not ok: cap.release(); raise RuntimeError("No frames.")
            if resize_w:
                h0,w0 = frame.shape[:2]
                frame = cv2.resize(frame, (resize_w, int(resize_w*h0/w0)))

            H,W = frame.shape[:2]
            excl = (0,0,W, int(H*exclude_top_pct))
            st.pitch_mask = build_pitch_mask(frame, exclude_rect=excl)
            bounds = estimate_pitch_bounds(st.pitch_mask)
            st.m_per_px  = estimate_m_per_px(st.pitch_mask)
            speed        = SpeedEstimator(m_per_px=st.m_per_px)

            
# ---- simple static UI mount ----
try:
    from fastapi.staticfiles import StaticFiles
    app.mount('/web', StaticFiles(directory='web', html=True), name='web')
except Exception:
    pass

# Serve minimal UI
app.mount("/web", StaticFiles(directory="web", html=True), name="web")
