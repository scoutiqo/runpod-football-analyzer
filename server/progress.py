# server/progress.py
import os
from fastapi import APIRouter, Request, Header, HTTPException
from server.sse_bus import BUS
router = APIRouter()
SECRET = os.getenv("CALLBACK_SECRET","changeme")
@router.post("/progress/{job_id}")
async def progress(job_id: str, request: Request, x_callback_secret: str = Header(None)):
    if x_callback_secret != SECRET:
        raise HTTPException(401, "bad secret")
    body = await request.json()
    print("PROGRESS", job_id, body)
    await BUS.publish(job_id, body)
    return {"ok": True}

