# handler.py (Fixed with Auto-Tuning + Duration)
import os, json, uuid, tempfile, time, traceback
from yt_dlp import YoutubeDL
import requests
import runpod
import cv2
from pipeline import run_pipeline
from evaluator import compute_metrics  # For auto-tuning
from ai_agent import decide, write_patch  # For auto-tuning

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_ROLE = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
ANALYSES_BUCKET = os.getenv("ANALYSES_BUCKET", "analyses")
CALLBACK_SECRET = os.getenv("CALLBACK_SECRET", "")

# ---------------- helpers ----------------
def sign_storage_path(bucket, path, expires=86400):
    url = f"{SUPABASE_URL}/storage/v1/object/sign/{bucket}/{path}"
    h = {"Authorization": f"Bearer {SERVICE_ROLE}", "Content-Type": "application/json"}
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

def post_cb(url, payload):
    """POST JSON to your FastAPI /progress/{job_id} callback WITH auth header."""
    try:
        headers = {
            "Content-Type": "application/json",
            "X-Callback-Token": CALLBACK_SECRET
        }
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if not r.ok:
            print(f"callback post failed {r.status_code}: {r.text[:400]}", flush=True)
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
            "connect_timeout": 300  # Reduced from 1200
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        for fname in os.listdir(tmpdir):
            if fname.lower().endswith(".mp4"):
                return os.path.join(tmpdir, fname)
        raise RuntimeError("yt-dlp did not produce an mp4 file.")
    r = requests.get(url, stream=True, timeout=300)  # Reduced timeout
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
    for attempt in range(3):  # Retry
        try:
            r = requests.post(url, headers=h, data=data, timeout=60)
            if r.ok:
                return
            print(f"Upload attempt {attempt+1} failed: {r.status_code}", flush=True)
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Upload attempt {attempt+1} error: {str(e)}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"upload failed after retries: {r.text}")

def upsert_rest(table, row, on_conflict):
    oc = ",".join(on_conflict) if isinstance(on_conflict, (list, tuple)) else str(on_conflict)
    url = f"{SUPABASE_URL}/rest/v1/{table}?on_conflict={oc}"
    headers = {
        "Authorization": f"Bearer {SERVICE_ROLE}",
        "apikey": SERVICE_ROLE,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation"
    }
    for attempt in range(3):  # Retry
        try:
            r = requests.post(url, headers=headers, data=json.dumps(row), timeout=60)
            if r.ok:
                return r.json() if r.text else {"status_code": r.status_code}
            print(f"Upsert attempt {attempt+1} failed: {r.status_code}", flush=True)
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Upsert attempt {attempt+1} error: {str(e)}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"upsert {table} failed after retries: {r.text}")

def handler(event):
    """
    Mode A (Realtime segments): {...}
    Mode B (Single video): {...}
    """
    inp = (event.get("input") or {})
    segs = inp.get("segment_urls")
    cb = inp.get("callback_url")
    # Load params for defaults
    cfg = json.load(open("params.json", "r", encoding="utf-8")) if os.path.exists("params.json") else {}

    # ---------- Mode A: realtime (segments + callback) ----------
    if segs and cb:
        simulate = bool(inp.get("simulate", True))
        max_frames = int(inp.get("max_frames", cfg.get("max_frames", 150)))
        frame_skip = int(inp.get("frame_skip", cfg.get("frame_skip", 2)))
        processed = 0
        all_tracks = []
        agg_meta = None
        t_offset = 0.0
        for i, url in enumerate(segs):
            post_cb(cb, {"type": "segment_start", "seg": i, "url": url})
            try:
                if simulate:
                    time.sleep(0.6)
                    metrics = {
                        "possession_pct": {"Home": 50 + (i % 5), "Away": 50 - (i % 5)},
                        "per_player": {"7": {"top_speed_mps": 8.5 + 0.1 * i}}
                    }
                    t_offset += 20.0
                else:
                    local = dl_tmp(url)
                    try:
                        duration_s = _video_duration_seconds(local)  # New: Log duration
                        out = run_pipeline(local, cfg=cfg, max_frames=max_frames, frame_skip=frame_skip)
                        if isinstance(out, dict) and not agg_meta:
                            agg_meta = out.get("meta", {}) or {}
                        seg_tracks = out.get("tracks", []) if isinstance(out, dict) else []
                        for r in seg_tracks:
                            rr = dict(r)
                            try:
                                rr["t"] = float(rr.get("t", 0.0)) + t_offset
                            except:
                                pass
                            all_tracks.append(rr)
                        seg_dur = duration_s or (max(float(z.get("t", 0.0)) for z in seg_tracks) if seg_tracks else 0.0)
                        t_offset += float(seg_dur)
                        metrics = compute_metrics(out)  # Real metrics
                        metrics["segment_index"] = i
                        metrics["duration_s"] = seg_dur
                    finally:
                        try:
                            if local and os.path.exists(local):
                                os.remove(local)
                        except:
                            pass
                post_cb(cb, {"type": "partial_metrics", "seg": i, **metrics})
                processed += 1
            except Exception as e:
                post_cb(cb, {"type": "error", "seg": i, "message": str(e)})
        # ---- assemble final artifact & upload to Supabase ----
        job_id = str(uuid.uuid4())[:8]  # Shortened for simplicity
        if simulate and not all_tracks:
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
                "metrics": compute_metrics({"tracks": all_tracks, "meta": agg_meta})  # Final metrics
            }
        # Auto-tuning for next run
        try:
            next_params = decide(cfg, result["metrics"])
            write_patch(".", "auto_patch.patch", next_params)
            try:
                from supabase import create_client
                supa = create_client(SUPABASE_URL, SERVICE_ROLE)
                supa.table("next_run_configs").insert({"created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), "params": next_params}).execute()
                result["auto_tune"] = {"next_params": next_params, "patch_written": "auto_patch.patch"}
            except Exception as e:
                print(f"Supabase push failed: {str(e)}", flush=True)
                result["auto_tune"] = {"error": str(e)}
        except Exception as e:
            print(f"Auto-tune failed: {str(e)}", flush=True)
            result["auto_tune"] = {"error": str(e)}
        key = f"jobs/{job_id}/tracks.json"
        upload_bytes(ANALYSES_BUCKET, key, json.dumps(result).encode("utf-8"), "application/json")
        tracks_url = sign_storage_path(ANALYSES_BUCKET, key, expires=86400)
        post_cb(cb, {
            "type": "done",
            "total_segments": len(segs),
            "processed": processed,
            "tracks_url": tracks_url,
            "auto_tune": result.get("auto_tune", {})
        })
        return {"ok": True, "mode": "segments", "processed": processed, "tracks_url": tracks_url, "auto_tune": result.get("auto_tune", {})}

    # ---------- Mode B: single-video Supabase pipeline ----------
    player_id = str(inp.get("player_id", "")).strip()
    video_url = str(inp.get("video_url", "")).strip()
    max_frames = int(inp.get("max_frames", cfg.get("max_frames", 150)))
    frame_skip = int(inp.get("frame_skip", cfg.get("frame_skip", 2)))
    if not player_id or not video_url:
        return {"ok": False, "error": "Either provide {segment_urls, callback_url} OR {player_id, video_url}"}
    job_id = str(uuid.uuid4())[:8]
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
        duration_s = _video_duration_seconds(local)  # New: Log duration
        print(f"Video duration: {duration_s:.2f}s", flush=True)
        result_tracks = run_pipeline(local, cfg=cfg, max_frames=max_frames, frame_skip=frame_skip)
        # Auto-tuning
        try:
            metrics = compute_metrics(result_tracks)
            next_params = decide(cfg, metrics)
            write_patch(".", "auto_patch.patch", next_params)
            try:
                from supabase import create_client
                supa = create_client(SUPABASE_URL, SERVICE_ROLE)
                supa.table("next_run_configs").insert({"created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), "params": next_params}).execute()
                result_tracks["auto_tune"] = {"next_params": next_params, "patch_written": "auto_patch.patch"}
            except Exception as e:
                print(f"Supabase push failed: {str(e)}", flush=True)
                result_tracks["auto_tune"] = {"error": str(e)}
        except Exception as e:
            print(f"Auto-tune failed: {str(e)}", flush=True)
            result_tracks["auto_tune"] = {"error": str(e)}
        key = f"players/{player_id}/{job_id}/tracks.json"
        upload_bytes(ANALYSES_BUCKET, key, json.dumps(result_tracks).encode("utf-8"), "application/json")
        tracks_url = sign_storage_path(ANALYSES_BUCKET, key, expires=86400)
        try:
            upsert_rest(
                "analysis_jobs",
                {
                    "id": job_id,
                    "player_id": player_id,
                    "status": "done",
                    "result_bucket": ANALYSES_BUCKET,
                    "result_path": key,
                    "result_url": tracks_url,
                    "duration_s": duration_s  # New
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
            print(f"WARN: upsert done failed: {str(e)}", flush=True)
        return {
            "ok": True,
            "mode": "single_video",
            "job_id": job_id,
            "bucket": ANALYSES_BUCKET,
            "path": key,
            "tracks_url": tracks_url,
            "duration_s": duration_s,  # New
            "auto_tune": result_tracks.get("auto_tune", {})
        }
    except Exception as e:
        tb = traceback.format_exc()
        try:
            upsert_rest(
                "analysis_jobs",
                {"id": job_id, "player_id": player_id, "status": "error"},
                on_conflict="id"
            )
        except:
            pass
        print("ERROR:", tb, flush=True)
        return {"ok": False, "mode": "single_video", "job_id": job_id, "error": str(e), "traceback": tb}
    finally:
        if local and os.path.exists(local):
            try:
                os.remove(local)
            except:
                pass

# Start serverless handler
runpod.serverless.start({"handler": handler})
