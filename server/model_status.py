# server/model_status.py
import os, json, time
from pathlib import Path

STORE = Path("./model_store")

def current_tags():
    """
    Get current model tags for all models
    """
    tags = {}
    for name in ["detector","xt","xg"]:
        p = STORE/f"{name}.txt"
        tags[name] = p.read_text().strip() if p.exists() else "default"
    return tags
