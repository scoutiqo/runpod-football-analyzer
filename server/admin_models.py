# server/admin_models.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pathlib

router = APIRouter(prefix="/admin")

class Switch(BaseModel):
    model: str
    tag: str

@router.post("/switch")
def switch(s: Switch):
    """Switch model version"""
    p = pathlib.Path("./model_store")/f"{s.model}.txt"
    p.write_text(s.tag)
    return {"ok": True, "model": s.model, "tag": s.tag}
