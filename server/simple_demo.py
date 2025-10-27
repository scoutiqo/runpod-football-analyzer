# server/simple_demo.py
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid, shutil, json, asyncio
from server.sse_bus import BUS

router = APIRouter()

@router.post("/demo/simple")
async def simple_demo(file: UploadFile = File(...), bg: BackgroundTasks = None):
    """
    Simple demo endpoint that just publishes SSE events without complex tracking
    """
    job = uuid.uuid4().hex[:8]
    jdir = Path("./files/jobs")/job
    jdir.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded file
    raw = jdir/"input.mp4"
    with raw.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Start background task
    if bg:
        bg.add_task(_simple_run, job, jdir)
    return JSONResponse({"job_id": job})

async def _simple_run(job: str, jdir: Path):
    """
    Simple background task that just publishes SSE events
    """
    try:
        print(f"[SIMPLE DEMO] Starting job {job}")
        await BUS.publish(job, {"type": "started"})
        
        # Simulate some processing time
        await asyncio.sleep(2)
        
        # Create dummy output files
        overlay_file = jdir / "overlay.mp4"
        tracks_file = jdir / "tracks.json"
        
        # Copy input to overlay (dummy)
        shutil.copy2(jdir / "input.mp4", overlay_file)
        
        # Create dummy tracks.json
        tracks_data = {
            "job_id": job,
            "players": [],
            "events": [],
            "metadata": {"duration": 0, "fps": 30}
        }
        with open(tracks_file, "w") as f:
            json.dump(tracks_data, f)
        
        await BUS.publish(job, {
            "type": "segment_done", 
            "seg": 0,
            "overlay_url": f"/files/jobs/{job}/overlay.mp4",
            "metrics_url": f"/files/jobs/{job}/tracks.json"
        })
        
        await asyncio.sleep(1)
        await BUS.publish(job, {"type": "done"})
        
        print(f"[SIMPLE DEMO] Job {job} completed successfully")
        
    except Exception as e:
        print(f"[SIMPLE DEMO] Error in job {job}: {e}")
        await BUS.publish(job, {"type": "segment_error", "error": str(e)})