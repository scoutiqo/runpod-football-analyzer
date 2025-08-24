# handler.py (UPDATED) — keeps your Supabase + DB logic
# Changes vs your current file:
# 1) After writing tracks.json, it now RETURNS a signed, fetchable URL (correct prefix /storage/v1).
# 2) Minor hardening + clearer error messages.

import traceback
from yt_dlp import YoutubeDL
import os, json, uuid, tempfile, requests
import runpod

from pipeline import run_pipeline  # our drop-in pipeline below

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_ROLE = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
ANALYSES_BUCKET = os.getenv("ANALYSES_BUCKET", "analyses")

# ---------------- helpers ----------------
def dl_tmp(url: str) -> str:
    """Download to a local temp .mp4. Supports YouTube and direct .mp4 URLs."""
    if "youtube.com" in url or "youtu.be" in url:
        tmpdir = tempfile.mkdtemp()
        outtmpl = os.path.join(tmpdir, "video.%(ext)s")
        ydl_opts = {
            "format": "bv*+ba/best",
            "merge_output_format": "mp4",
            "outtmpl": outtmpl,
            "quiet": True,
            "noprogress": True,
            "extractor_args": {"youtube": {"player_client": ["android"]}},
            "retries": 5,
            "fragment_retries": 5,
            "concurrent_fragment_downloads": 1,
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        for fname in os.listdir(tmpdir):
            if fname.lower().endswith(".mp4"):
                return os.path.join(tmpdir, fname)
        raise RuntimeError("yt-dlp did not produce an mp4 file.")
    # direct file download
    r = requests.get(url, stream=True, timeout=1200)
    r.raise_for_status()
    fd, p = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    with open(p, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    return p

def upload_bytes(bucket, path, data, content_type):
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    h = {
        "Authorization": f"Bearer {SERVICE_ROLE}",
        "x-upsert": "true",
        "Content-Type": content_type
    }
    r = requests.post(url, headers=h, data=data, timeout=1200)
    if not r.ok:
        raise RuntimeError(f"upload failed {r.status_code}: {r.text}")

def post_rest(table, row):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "Authorization": f"Bearer {SERVICE_ROLE}",
        "apikey": SERVICE_ROLE,
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    r = requests.post(url, headers=headers, data=json.dumps(row), timeout=60)
    if not r.ok:
        raise RuntimeError(f"post {table} failed {r.status_code}: {r.text}")
    return r.json() if r.text else {"status_code": r.status_code}

def upsert_rest(table, row, on_conflict):
    oc = ",".join(on_conflict) if isinstance(on_conflict, (list, tuple)) else str(on_conflict)
    url = f"{SUPABASE_URL}/rest/v1/{table}?on_conflict={oc}"
    headers = {
        "Authorization": f"Bearer {SERVICE_ROLE}",
        "apikey": SERVICE_ROLE,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation"
    }
    r = requests.post(url, headers=headers, data=json.dumps(row), timeout=60)
    if not r.ok:
        raise RuntimeError(f"upsert {table} failed {r.status_code}: {r.text}")
    return r.json() if r.text else {"status_code": r.status_code}

def sign_storage_path(bucket, path, expires=86400):
    """Return absolute signed URL with the correct /storage/v1 prefix."""
    url = f"{SUPABASE_URL}/storage/v1/object/sign/{bucket}/{path}"
    h = { "Authorization": f"Bearer {SERVICE_ROLE}", "Content-Type": "application/json" }
    r = requests.post(url, headers=h, data=json.dumps({"expiresIn": int(expires)}), timeout=30)
    if not r.ok:
        raise RuntimeError(f"sign failed {r.status_code}: {r.text}")
    data = r.json()
    signed = data.get("signedURL") or data.get("signedUrl")
    if not signed:
        raise RuntimeError(f"sign response missing signedURL: {data}")
    # IMPORTANT: prepend /storage/v1
    return f"{SUPABASE_URL}/storage/v1{signed}"

# ---------------- runpod handler ----------------
def handler(event):
    inp = event.get("input", {}) or {}
    player_id  = str(inp.get("player_id", "")).strip()
    video_url  = str(inp.get("video_url", "")).strip()
    max_frames = int(inp.get("max_frames", 150))
    frame_skip = int(inp.get("frame_skip", 2))

    if not player_id or not video_url:
        return {"ok": False, "error": "player_id and video_url required"}

    job_id = str(uuid.uuid4())

    # mark RUNNING
    try:
        upsert_rest(
            "analysis_jobs",
            {"id": job_id, "player_id": player_id, "status": "running", "video_url": video_url},
            on_conflict="id"
        )
    except Exception as e:
        # proceed even if table not configured yet
        print("WARN: analysis_jobs upsert running failed:", e, flush=True)

    local = None
    try:
        local = dl_tmp(video_url)

        # ---- RUN YOUR PIPELINE (now guaranteed to return non-empty tracks if people are detected)
        result_tracks = run_pipeline(local, max_frames=max_frames, frame_skip=frame_skip)

        key = f"players/{player_id}/{job_id}/tracks.json"
        upload_bytes(ANALYSES_BUCKET, key, json.dumps(result_tracks).encode("utf-8"), "application/json")

        tracks_url = sign_storage_path(ANALYSES_BUCKET, key, expires=86400)

        # mark DONE
        try:
            upsert_rest(
                "analysis_jobs",
                {
                    "id": job_id,
                    "player_id": player_id,
                    "status": "done",
                    "result_bucket": ANALYSES_BUCKET,
                    "result_path": key,
                    "result_url": tracks_url
                },
                on_conflict="id"
            )
            upsert_rest(
                "player_latest_analysis",
                {
                    "player_id": player_id,
                    "analysis_job_id": job_id,
                    "result_bucket": ANALYSES_BUCKET,
                    "result_path": key,
                    "result_url": tracks_url
                },
                on_conflict="player_id"
            )
        except Exception as e:
            print("WARN: upsert done failed:", e, flush=True)

        return {"ok": True, "job_id": job_id, "bucket": ANALYSES_BUCKET, "path": key, "tracks_url": tracks_url}

    except Exception as e:
        tb = traceback.format_exc()
        try:
            upsert_rest(
                "analysis_jobs",
                {"id": job_id, "player_id": player_id, "status": "error"},
                on_conflict="id"
            )
        except Exception:
            pass
        print("ERROR:", tb, flush=True)
        return {"ok": False, "job_id": job_id, "error": str(e), "traceback": tb}

    finally:
        if local and os.path.exists(local):
            try: os.remove(local)
            except Exception: pass

runpod.serverless.start({"handler": handler})
