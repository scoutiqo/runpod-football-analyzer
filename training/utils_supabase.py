# /training/utils_supabase.py
import os
import mimetypes
import requests
from pathlib import Path
from typing import List

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE")
DEFAULT_BUCKET = os.environ.get("ANALYSES_BUCKET", "analyses")

def _headers():
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY missing")
    return {"apikey": SUPABASE_SERVICE_ROLE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}"}

def upload_file(local_path: Path, bucket: str, dest_path: str):
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{dest_path}"
    ctype, _ = mimetypes.guess_type(str(local_path))
    with open(local_path, "rb") as f:
        r = requests.put(url, headers=_headers(), data=f if ctype is None else f, params={})
        # Note: Supabase Storage PUT ignores Content-Type; safe to omit.
    if r.status_code not in (200, 201, 204):
        raise RuntimeError(f"Upload failed {r.status_code}: {r.text}")

def upload_folder(local_dir: Path, bucket: str, dest_prefix: str) -> List[str]:
    uploaded = []
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(local_dir)).replace("\\", "/")
            dst = f"{dest_prefix}/{rel}"
            upload_file(p, bucket, dst)
            uploaded.append(f"{bucket}/{dst}")
    return uploaded
