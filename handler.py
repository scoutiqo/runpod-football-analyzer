# handler.py (COMBINED, upgraded)
# - Mode A (Realtime): accepts segment_urls + callback_url, posts live progress,
#   merges tracks across segments with time offsets, uploads tracks.json to Supabase,
#   and includes tracks_url in the final "done" event.
# - Mode B (Existing): accepts video_url + player_id, runs your pipeline, uploads to Supabase.

import os, json, uuid, tempfile, time, traceback
from yt_dlp import YoutubeDL
import requests
import runpod
import cv2  # for segment duration

from pipeline import run_pipeline  # your existing pipeline

SUPABASE_URL    = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_ROLE    = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
ANALYSES_BUCKET = os.getenv("ANALYSES_BUCKET", "analyses")

# ---------------- helpers ----------------
def post_cb(url, payload):
    """POST JSON to your FastAPI /progress/{job_id} callback."""
    try:
        requests.post(url, json=payload, timeout=30)
    except Exception as e:
        print("callback failed:", e, flush=True)

def dl_tmp(url: str) -> str:
    """Download to a local temp .mp4. Supports YouTube and direct/signed .mp4 URLs."""
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

    # direct/signed URL download
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
    url = f"{SUPABASE_URL}/storage/v1/object/sign/{bucket}/{path}"
    h = { "Authorization": f"Bearer {SERVICE_ROLE}", "Content-Type": "application/json" }
    r = requests.post(url, headers=h, data=json.dumps({"expiresIn": int(expires)}), timeout=30)
    if not r.ok:
        raise RuntimeError(f"sign failed {r.status_code}: {r.text}")
    data = r.json()
    signed = data.get("signedURL") or data.get("signedUrl")
    if not signed:
        raise RuntimeError(f"sign response missing signedURL: {data}")
    return f"{SUPABASE_URL}/storage/v1{signed}"

def _video_duration_seconds(local_path: str) -> float:
    """Return duration in seconds using OpenCV; robust to missing metadata."""
    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        return 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    try:
        return float(frames) / float(fps) if frames and fps else 0.0
    except Exception:
        return 0.0

# ---------------- runpod handler ----------------
def handler(event):
    """
    Mode A (Realtime segments):
      input = {
        "segment_urls": ["https://.../seg_000.mp4", ...],
        "callback_url": "https://<PUBLIC_BASE_URL>/progress/<job_id>",
        "simulate": true,                # default true -> quick progress
        "max_frames": 150, "frame_skip": 2
      }

    Mode B (Existing single video -> Supabase):
      input = {
        "player_id": "uuid-or-id",
        "video_url": "https://...mp4 | youtube url",
        "max_frames": 150, "frame_skip": 2
      }
    """
    inp = (event.get("input") or {})
    segs = inp.get("segment_urls")
    cb   = inp.get("callback_url")

    # ---------- Mode A: realtime (segments + callback) ----------
    if segs and cb:
        simulate   = bool(inp.get("simulate", True))
        max_frames = int(inp.get("max_frames", 150))
        frame_skip = int(inp.get("frame_skip", 2))

        processed = 0
        all_tracks = []
        agg_meta = None
        t_offset = 0.0  # seconds

        for i, url in enumerate(segs):
            post_cb(cb, {"type": "segment_start", "seg": i, "url": url})
            try:
                if simulate:
                    # Quick, guaranteed progress for end-to-end wiring tests.
                    time.sleep(0.6)
                    # Simulated metrics
                    metrics = {
                        "possession_pct": {"Home": 50 + (i % 5), "Away": 50 - (i % 5)},
                        "per_player": {"7": {"top_speed_mps": 8.5 + 0.1 * i}}
                    }
                    # Assume ~20s per segment when simulating.
                    t_offset += 20.0
                else:
                    # Real processing per segment
                    local = dl_tmp(url)
                    try:
                        out = run_pipeline(local, max_frames=max_frames, frame_skip=frame_skip)
                        if isinstance(out, dict) and not agg_meta:
                            agg_meta = out.get("meta", {}) or {}
                        seg_tracks = out.get("tracks", []) if isinstance(out, dict) else []

                        # Shift segment times by t_offset, append to global list
                        for r in seg_tracks:
                            rr = dict(r)
                            try:
                                rr["t"] = float(rr.get("t", 0.0)) + t_offset
                            except Exception:
                                pass
                            all_tracks.append(rr)

                        # Use actual duration if we can, else fallback to last t
                        seg_dur = _video_duration_seconds(local)
                        if seg_dur <= 0.0:
                            if seg_tracks:
                                seg_dur = max(float(z.get("t", 0.0)) for z in seg_tracks)
                        t_offset += float(seg_dur or 0.0)

                        metrics = {"tracks_found": len(seg_tracks), "segment_index": i}
                    finally:
                        try:
                            if local and os.path.exists(local):
                                os.remove(local)
                        except Exception:
                            pass

                post_cb(cb, {"type": "partial_metrics", "seg": i, **metrics})
                processed += 1
            except Exception as e:
                post_cb(cb, {"type": "error", "seg": i, "message": str(e)})

        # ---- assemble final artifact & upload to Supabase ----
        job_id = str(uuid.uuid4())
        if simulate and not all_tracks:
            # produce a valid empty artifact for simulate mode
            result = {
                "version": 2,
                "meta": {"pitch_m": [105, 68], "note": "simulated"},
                "tracks": [],
                "events": [],
                "metrics": {}
            }
        else:
            result = {
                "version": 2,
                "meta": agg_meta or {"pitch_m": [105, 68], "note": "aggregated"},
                "tracks": sorted(all_tracks, key=lambda z: float(z.get("t", 0.0))),
                "events": [],
                "metrics": {}
            }

        key = f"jobs/{job_id}/tracks.json"
        upload_bytes(ANALYSES_BUCKET, key, json.dumps(result).encode("utf-8"), "application/json")
        tracks_url = sign_storage_path(ANALYSES_BUCKET, key, expires=86400)

        post_cb(cb, {
            "type": "done",
            "total_segments": len(segs),
            "processed": processed,
            "tracks_url": tracks_url
        })
        return {"ok": True, "mode": "segments", "processed": processed, "tracks_url": tracks_url}

    # ---------- Mode B: your existing single-video Supabase pipeline ----------
    player_id  = str(inp.get("player_id", "")).strip()
    video_url  = str(inp.get("video_url", "")).strip()
    max_frames = int(inp.get("max_frames", 150))
    frame_skip = int(inp.get("frame_skip", 2))

    if not player_id or not video_url:
        return {"ok": False, "error": "Either provide {segment_urls, callback_url} OR {player_id, video_url}"}

    job_id = str(uuid.uuid4())

    # mark RUNNING (best effort)
    try:
        upsert_rest(
            "analysis_jobs",
            {"id": job_id, "player_id": player_id, "status": "running", "video_url": video_url},
            on_conflict="id"
        )
    except Exception as e:
        print("WARN: analysis_jobs upsert running failed:", e, flush=True)

    local = None
    try:
        local = dl_tmp(video_url)

        # run your pipeline
        result_tracks = run_pipeline(local, max_frames=max_frames, frame_skip=frame_skip)

        key = f"players/{player_id}/{job_id}/tracks.json"
        upload_bytes(ANALYSES_BUCKET, key, json.dumps(result_tracks).encode("utf-8"), "application/json")
        tracks_url = sign_storage_path(ANALYSES_BUCKET, key, expires=86400)

        # mark DONE (best effort)
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

        return {"ok": True, "mode": "single_video", "job_id": job_id, "bucket": ANALYSES_BUCKET, "path": key, "tracks_url": tracks_url}

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
        return {"ok": False, "mode": "single_video", "job_id": job_id, "error": str(e), "traceback": tb}

    finally:
        if local and os.path.exists(local):
            try:
                os.remove(local)
            except Exception:
                pass

# Start serverless handler
runpod.serverless.start({"handler": handler})
