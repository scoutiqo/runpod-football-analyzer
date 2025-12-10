# server/admin_readouts.py
from fastapi import APIRouter
from pathlib import Path
import json, time

router = APIRouter(prefix="/admin")

@router.get("/models")
def models():
    """
    Get current model tags
    """
    from server.model_status import current_tags
    return current_tags()

@router.get("/metrics")
def metrics():
    """
    Get latest training metrics
    """
    p = Path("./model_store/metrics.json")
    return json.loads(p.read_text()) if p.exists() else {}
