# server/live_track_demo.py  (local, single-shot proof)
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid, shutil, json, cv2
from server.sse_bus import BUS
from analyzers.tracking import run_tracking, TrackerCfg
router = APIRouter()
@router.post("/demo/track")
async def demo_track(file: UploadFile = File(...), bg: BackgroundTasks = None):
    job = uuid.uuid4().hex[:8]
    jdir = Path("./files/jobs")/job; jdir.mkdir(parents=True, exist_ok=True)
    raw = jdir/"input.mp4";  raw.write_bytes(await file.read())
    if bg: bg.add_task(_run, job, raw, jdir)
    return JSONResponse({"job_id": job})
async def _run(job, raw: Path, jdir: Path):
    await BUS.publish(job, {"type":"started"})
    tracks_json, overlay_mp4 = run_tracking(str(raw), str(jdir), TrackerCfg())
    Path(overlay_mp4).replace(jdir/"overlay.mp4")
    Path(tracks_json).replace(jdir/"tracks.json")
    await BUS.publish(job, {"type":"segment_done","seg":0,
                            "overlay_url": f"/files/jobs/{job}/overlay.mp4",
                            "metrics_url": f"/files/jobs/{job}/tracks.json"})
    await BUS.publish(job, {"type":"done"})