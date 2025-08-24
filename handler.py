# handler.py
import traceback
from yt_dlp import YoutubeDL
import os, json, uuid, tempfile, requests
import runpod

from pipeline import run_pipeline  # your pipeline

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_ROLE = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
ANALYSES_BUCKET = os.getenv("ANALYSES_BUCKET", "analyses")

# ---------- helpers ----------
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
            # helps with some interstitials; YT may still block -> prefer direct MP4/signed URL
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
    r = requests.get(url, stream=True, timeout=60)
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
    r = requests.post(url, headers=h, data=data, timeout=120)
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
    r = requests.post(url, headers=headers, data=json.dumps(row), timeout=30)
    if not r.ok:
        raise RuntimeError(f"post {table} failed {r.status_code}: {r.text}")
    return r.json() if r.text else {"status_code": r.status_code}


def upsert_rest(table, row, on_conflict):
    """INSERT ... ON CONFLICT DO UPDATE via PostgREST."""
    oc = ",".join(on_conflict) if isinstance(on_conflict, (list, tuple)) else str(on_conflict)
    url = f"{SUPABASE_URL}/rest/v1/{table}?on_conflict={oc}"
    headers = {
        "Authorization": f"Bearer {SERVICE_ROLE}",
        "apikey": SERVICE_ROLE,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation"
    }
    r = requests.post(url, headers=headers, data=json.dumps(row), timeout=30)
    if not r.ok:
        raise RuntimeError(f"upsert {table} failed {r.status_code}: {r.text}")
    return r.json() if r.text else {"status_code": r.status_code}


def patch_rest(table, where, row):
    url = f"{SUPABASE_URL}/rest/v1/{table}?{where}"
    headers = {
        "Authorization": f"Bearer {SERVICE_ROLE}",
        "apikey": SERVICE_ROLE,
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    r = requests.patch(url, headers=headers, data=json.dumps(row), timeout=30)
    if not r.ok:
        raise RuntimeError(f"patch {table} failed {r.status_code}: {r.text}")
    return r.json() if r.text else {"status_code": r.status_code}


# ---------- runpod handler ----------
def handler(event):
    inp = event.get("input", {}) or {}
    player_id  = str(inp.get("player_id", "")).strip()
    video_url  = str(inp.get("video_url", "")).strip()
    max_frames = int(inp.get("max_frames", 150))
    frame_skip = int(inp.get("frame_skip", 2))

    if not player_id or not video_url:
        return {"ok": False, "error": "player_id and video_url required"}

    job_id = str(uuid.uuid4())

    # Create job row (no created_at)
    post_rest("analysis_jobs", {
        "id": job_id,
        "player_id": player_id,
        "status": "running",
        "video_url": video_url
    })

    local = None
    try:
        local = dl_tmp(video_url)
        result = run_pipeline(local, max_frames=max_frames, frame_skip=frame_skip)

        key = f"players/{player_id}/{job_id}/tracks.json"
        upload_bytes(ANALYSES_BUCKET, key, json.dumps(result).encode("utf-8"), "application/json")

        # Mark job done (no finished_at)
        patch_rest(
            "analysis_jobs",
            f"id=eq.{job_id}",
            {"status": "done", "result_bucket": ANALYSES_BUCKET, "result_path": key}
        )

        # Keep latest pointer with UPSERT (avoids 409 duplicates)
        upsert_rest(
            "player_latest_analysis",
            {
                "player_id": player_id,
                "analysis_job_id": job_id,
                "result_bucket": ANALYSES_BUCKET,
                "result_path": key
            },
            on_conflict="player_id"
        )

        return {"ok": True, "job_id": job_id, "bucket": ANALYSES_BUCKET, "path": key}

    except Exception as e:
        tb = traceback.format_exc()
        try:
            patch_rest("analysis_jobs", f"id=eq.{job_id}", {"status": "error", "error": str(e)})
        except Exception:
            pass
        print("ERROR:", tb, flush=True)
        return {"ok": False, "job_id": job_id, "error": str(e), "traceback": tb}

    finally:
        if local and os.path.exists(local):
            try:
                os.remove(local)
            except Exception:
                pass


runpod.serverless.start({"handler": handler})
