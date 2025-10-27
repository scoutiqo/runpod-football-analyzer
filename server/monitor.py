# server/monitor.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from server.sse_bus import BUS
import json
router = APIRouter()
@router.get("/monitor/{job_id}")
async def monitor(job_id: str):
    async def gen():
        yield b"retry: 2000\n"; yield b": keepalive\n\n"
        q = await BUS.subscribe(job_id)
        try:
            while True:
                m = await q.get()
                yield b"data: " + json.dumps(m).encode() + b"\n\n"
        finally:
            await BUS.unsubscribe(job_id, q)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

