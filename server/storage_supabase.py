# server/storage_supabase.py
import os
import time
import mimetypes
from pathlib import Path
import requests

SUPABASE_URL  = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
BUCKET_NAME   = os.environ.get("SUPABASE_BUCKET", "scoutiqo")  # change if you use a different bucket

if not SUPABASE_URL or not SUPABASE_KEY:
    # We'll only error at runtime if methods are actually called.
    pass

# NOTE: storage REST is under /storage/v1
BASE = f"{SUPABASE_URL}/storage/v1"
HDRS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}

def _guess_ct(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"

def put_file(local_path: str, object_key: str) -> dict:
    """
    Upload local file to Supabase bucket at object_key.
    Returns a dict with:
      {"bucket": str, "key": str, "signed_url": str, "public_url": str}
    """
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(local_path)

    ct = _guess_ct(str(p))
    url = f"{BASE}/object/{BUCKET_NAME}/{object_key}"
    with open(p, "rb") as f:
        r = requests.post(url, headers={**HDRS, "Content-Type": ct}, data=f)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase upload failed {r.status_code}: {r.text[:400]}")

    # Signed URL (default 24h)
    signed = sign_url(object_key, expires_in=24 * 3600)

    # Public URL (if bucket is public; safe to return either way)
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{object_key}"
    return {"bucket": BUCKET_NAME, "key": object_key, "signed_url": signed, "public_url": public_url}

def sign_url(object_key: str, expires_in: int = 3600) -> str:
    """
    Create a time-limited signed URL for the given object.
    """
    url = f"{BASE}/object/sign/{BUCKET_NAME}/{object_key}"
    r = requests.post(url, headers=HDRS, json={"expiresIn": expires_in})
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase sign failed {r.status_code}: {r.text[:400]}")
    # API returns {"signedURL": "..."} (path relative to /storage/v1)
    signed_path = r.json().get("signedURL") or r.json().get("signedUrl")
    if not signed_path:
        raise RuntimeError(f"Supabase sign response missing signedURL: {r.text[:400]}")
    # Return an absolute URL
    return f"{SUPABASE_URL}/storage/v1{signed_path}"

def ensure_bucket_exists() -> None:
    """
    No-op here (assume bucket already exists).
    If you need, you can create/manage via Supabase Dashboard.
    """
    return
