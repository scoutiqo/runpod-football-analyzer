# server/rescue_local.py
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid, shutil, json, cv2
from sse_bus import BUS  # your bus module
from analyzers.tracking import run_tracking, TrackerCfg
from phoenix.pipeline import phoenix_run_segment, PhoenixCfg

router = APIRouter()

@router.post("/e2e")
async def e2e(file: UploadFile = File(...), bg: BackgroundTasks = None):
    """
    End-to-end processing with real-time SSE updates
    """
    job = uuid.uuid4().hex[:8]
    jdir = Path("./files/jobs")/job
    jdir.mkdir(parents=True, exist_ok=True)
    raw = jdir/"input.mp4"
    with raw.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    # kick background worker
    bg.add_task(_run_job, job, raw, jdir)
    return JSONResponse({"job_id": job})

async def _run_job(job: str, raw: Path, jdir: Path):
    """
    Background job processor with real-time updates
    """
    await BUS.publish(job, {"type":"started"})
    
    # pass 1: tracking → overlay + tracks.json
    tracks_json, overlay_mp4 = run_tracking(str(raw), str(jdir), TrackerCfg())
    # expose with predictable names
    Path(overlay_mp4).replace(jdir/"overlay.mp4")
    Path(tracks_json).replace(jdir/"tracks.json")
    await BUS.publish(job, {
        "type":"segment_done","seg":0,
        "overlay_url": f"/files/jobs/{job}/overlay.mp4",
        "metrics_url": f"/files/jobs/{job}/tracks.json"
    })

    # pass 2: phoenix enrich → events_phoenix.json
    cap = cv2.VideoCapture(str(raw))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # Re-use run_tracking output to drive phoenix
    TJ = json.loads((jdir/"tracks.json").read_text())
    # build iterators
    def frames():
        c = cv2.VideoCapture(str(raw))
        while True:
            ok, fr = c.read()
            if not ok: break
            yield fr
        c.release()
    def det_stream():
        while True:
            yield {"ball": None}
    def track_stream():
        per_f={}
        for p in TJ.get("players", []):
            for fr in p["frames"]:
                per_f.setdefault(fr["f"], {})[p["tid"]] = [0,0,0,0]
        f=0
        while True:
            yield per_f.get(f, {})
            f += 1

    res = phoenix_run_segment(frames(), det_stream(), track_stream(), (W,H), PhoenixCfg(fps=fps))
    nodes=[dict(i=n.i,t=n.t,team=n.team,kind=n.kind,actor=n.actor,x=n.xy[0],y=n.xy[1],tags=n.tags) for n in res["graph"].nodes]
    edges=[dict(u=e.u,v=e.v,dt=e.dt,dx=e.dx,dy=e.dy,value=e.value) for e in res["graph"].edges]
    (jdir/"events_phoenix.json").write_text(json.dumps({"nodes":nodes,"edges":edges}))
    await BUS.publish(job, {"type":"done"})
