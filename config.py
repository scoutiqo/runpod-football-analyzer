# config.py
import os, requests

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_ROLE = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

def fetch_config(config_id="default"):
    url = f"{SUPABASE_URL}/rest/v1/pipeline_config?id=eq.{config_id}&select=config"
    h = {"Authorization": f"Bearer {SERVICE_ROLE}", "apikey": SERVICE_ROLE}
    r = requests.get(url, headers=h, timeout=15)
    if r.ok and r.json():
        return r.json()[0]["config"]
    # fallback defaults
    return {
        "detector": {"backend":"yolov8","weights_bucket":"models","weights_path":"prod/best.pt","conf":0.25,"iou":0.45},
        "tracking": {"max_age":30,"min_hits":3},
        "ball": {"class_id":32,"min_conf":0.20},
        "frame":{"frame_skip":3,"max_frames":6000}
    }

def sign_storage_url(bucket, path, expires=3600):
    url = f"{SUPABASE_URL}/storage/v1/object/sign/{bucket}/{path}"
    h = {"Authorization": f"Bearer {SERVICE_ROLE}"}
    r = requests.post(url, headers=h, json={"expiresIn": expires}, timeout=15)
    r.raise_for_status()
    signed = r.json().get("signedURL") or r.json().get("signedUrl")
    return f"{SUPABASE_URL}/storage/v1{signed}"
