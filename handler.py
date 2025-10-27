# handler.py (Updated with new pipeline)
import os, json, uuid, tempfile, time, traceback
from pathlib import Path
from yt_dlp import YoutubeDL
import requests
import runpod
import cv2
import numpy as np
from pipeline import run_pipeline
from evaluator import compute_metrics  # For auto-tuning
from ai_agent import decide, write_patch  # For auto-tuning

# Import new modules
from analyzers.tracking import run_tracking
from analyzers.events import extract_events
from analyzers.value_models import compute_values
from metrics.players import aggregate_player_metrics
from render.heatmap import player_heatmap
from merge.segments import merge_all, SegmentMerger
from schemas.contracts import TracksJSON, PlayerInsights

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
    Updated handler with new pipeline architecture
    
    Processes video segments through:
    1. Tracking (YOLO + ByteTrack)
    2. Event detection
    3. Value models (xT, EPV, VAEP)
    4. Player metrics aggregation
    5. Heatmap generation
    6. Segment merging
    """
    inp = (event.get("input") or {})
    job_id = inp.get("job_id")
    segs = inp.get("segment_urls")
    cb = inp.get("callback_url")
    callback_secret = inp.get("callback_secret")
    
    # Load params for defaults
    cfg = json.load(open("params.json", "r", encoding="utf-8")) if os.path.exists("params.json") else {}

    # ---------- Mode A: realtime (segments + callback) ----------
    if segs and cb and job_id:
        simulate = bool(inp.get("simulate", False))
        processed = 0
        segment_results = []
        
        for i, url in enumerate(segs):
            post_cb(cb, {"type": "segment_start", "seg": i, "url": url})
            try:
                if simulate:
                    # Simulate processing
                    time.sleep(0.6)
                    segment_result = {
                        "segment_idx": i,
                        "players": [],
                        "events": [],
                        "artifacts": {"overlays": [], "logs": []},
                        "auto_tune": {}
                    }
                else:
                    # Real processing
                    local = dl_tmp(url)
                    try:
                        # Run real tracking with YOLO + ByteTrack
                        from analyzers.tracking import TrackerCfg
                        cfg_tracking = TrackerCfg()
                        tracks_json_path, overlay_mp4_path = run_tracking(local, f"/tmp/jobs/{job_id}/seg_{i:03d}", cfg_tracking)
                        
                        # Load tracking results
                        with open(tracks_json_path, 'r') as f:
                            tracking_result = json.load(f)
                        
                        # Run Phoenix pipeline for advanced analytics
                        try:
                            from phoenix.pipeline import phoenix_run_segment, PhoenixCfg
                            from phoenix.serializer import save_json_serializable_graph
                            import cv2
                            
                            # Create frame iterator
                            cap = cv2.VideoCapture(local)
                            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            def frames_iter():
                                while True:
                                    ret, frame = cap.read()
                                    if not ret: break
                                    yield frame
                                cap.release()
                            
                            # Create detection stream (simplified - would use actual YOLO detections)
                            def det_stream():
                                for ball_data in tracking_result.get("ball", []):
                                    if ball_data["f"] < len(tracking_result.get("ball", [])):
                                        yield {"ball": (ball_data["cx"] * W, ball_data["cy"] * H, 
                                                       ball_data["cx"] * W + 10, ball_data["cy"] * H + 10)}
                                    else:
                                        yield {"ball": None}
                            
                            # Create tracking stream
                            def track_stream():
                                for f in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                                    frame_tracks = {}
                                    for player_data in tracking_result.get("players", []):
                                        for frame_data in player_data["frames"]:
                                            if frame_data["f"] == f:
                                                frame_tracks[player_data["tid"]] = [
                                                    frame_data["cx"] * W - frame_data["w"] * W / 2,
                                                    frame_data["cy"] * H - frame_data["h"] * H / 2,
                                                    frame_data["cx"] * W + frame_data["w"] * W / 2,
                                                    frame_data["cy"] * H + frame_data["h"] * H / 2
                                                ]
                                    yield frame_tracks
                            
                            # Run Phoenix pipeline
                            phoenix_result = phoenix_run_segment(
                                frames_iter(), 
                                det_stream(), 
                                track_stream(), 
                                (W, H), 
                                PhoenixCfg(fps=fps, attack_dir={"home": +1, "away": -1})
                            )
                            
                        # Save Phoenix results
                        phoenix_paths = save_json_serializable_graph(phoenix_result, f"/tmp/jobs/{job_id}/seg_{i:03d}")
                        
                        # Collect data for training
                        from automl.collector import collect_job
                        collect_job(job_id, f"/tmp/jobs/{job_id}/seg_{i:03d}")
                        
                        print(f"Phoenix pipeline completed for segment {i}")
                            
                        except Exception as e:
                            print(f"Phoenix pipeline failed for segment {i}: {e}")
                            # Continue with regular processing
                        
                        # Build FrameState list for advanced analytics
                        from events.possessions import build_possessions, FrameState
                        from events.actions import detect_actions
                        from metrics.enrich import enrich_events
                        from metrics.pitch import px_to_pitch, PitchCalib
                        
                        # Convert tracking data to FrameState format
                        frames = []
                        fps = tracking_result.get("video", {}).get("fps", 30.0)
                        W = tracking_result.get("video", {}).get("width", 1920)
                        H = tracking_result.get("video", {}).get("height", 1080)
                        
                        # Create dummy team assignment (alternating)
                        tids = [p["tid"] for p in tracking_result.get("players", [])]
                        team_of = {tid: ("home" if i%2==0 else "away") for i, tid in enumerate(tids)}
                        
                        # Build frame states
                        max_frame = 0
                        if tracking_result.get("players"):
                            max_frame = max(max(fr["f"] for pl in tracking_result["players"] for fr in pl["frames"]), 
                                          max(fr["f"] for fr in tracking_result.get("ball", [])))
                        
                        # Create ball and player position maps
                        ball_positions = {fr["f"]: (fr["cx"], fr["cy"]) for fr in tracking_result.get("ball", [])}
                        player_positions = {}
                        for pl in tracking_result.get("players", []):
                            player_positions[pl["tid"]] = {fr["f"]: (fr["cx"], fr["cy"]) for fr in pl["frames"]}
                        
                        # Convert normalized coordinates to meters
                        calib = PitchCalib()  # Using fallback normalization
                        
                        for f in range(max_frame + 1):
                            if f in ball_positions:
                                ball_xy_norm = np.array([ball_positions[f]])
                                ball_xy_m = px_to_pitch(ball_xy_norm, calib, (W, H))[0]
                                
                                player_xy_m = {}
                                for tid in tids:
                                    if tid in player_positions and f in player_positions[tid]:
                                        player_xy_norm = np.array([player_positions[tid][f]])
                                        player_xy_m[tid] = px_to_pitch(player_xy_norm, calib, (W, H))[0]
                                
                                frames.append(FrameState(
                                    f=f,
                                    ball_xy=tuple(ball_xy_m),
                                    player_xy=player_xy_m,
                                    team_of=team_of
                                ))
                        
                        # Build possessions and detect events
                        poss = build_possessions(frames)
                        events = detect_actions(frames, fps, poss, goal_x=105.0)
                        
                        # Enrich events with values
                        frames_by_f = {s.f: s for s in frames}
                        frames_by_f["fps"] = fps
                        enriched_events = enrich_events(events, frames_by_f, team_of, attack_dir={"home":+1,"away":-1})
                        
                        # Save events alongside tracks
                        events_path = os.path.join(f"/tmp/jobs/{job_id}/seg_{i:03d}", "events.json")
                        with open(events_path, 'w') as f:
                            json.dump({
                                "possessions": poss,
                                "events": [e for e in map(dict, enriched_events)]
                            }, f)
                        
                        # Generate heatmaps for players
                        player_heatmaps = {}
                        unique_players = set()
                        for event in enriched_events:
                            unique_players.add(event.actor_tid)
                        
                        for player_id in unique_players:
                            heatmap_path = player_heatmap(tracking_result, player_id)
                            player_heatmaps[player_id] = heatmap_path
                        
                        # Aggregate player metrics
                        player_metrics = aggregate_player_metrics(tracking_result, enriched_events)
                        
                        # Create segment result
                        segment_result = {
                            "segment_idx": i,
                            "players": [
                                {
                                    "tid": tid,
                                    "team": "unknown",  # Would need team assignment
                                    "jersey": None,
                                    "role_hint": None,
                                    "primary_position": None,
                                    "metrics": metrics.dict(),
                                    "heatmap": player_heatmaps.get(tid, ""),
                                    "events_idx": [j for j, e in enumerate(enriched_events) if e.actor_tid == tid]
                                }
                                for tid, metrics in player_metrics.items()
                            ],
                            "events": [event.dict() for event in enriched_events],
                            "artifacts": {
                                "overlays": [],  # Would generate overlay videos
                                "logs": []
                            },
                            "auto_tune": {}
                        }
                        
                    finally:
                        try:
                            if local and os.path.exists(local):
                                os.remove(local)
                        except:
                            pass
                
                segment_results.append(segment_result)
                post_cb(cb, {"type": "segment_done", "seg": i, "metrics": segment_result})
                processed += 1
                
            except Exception as e:
                post_cb(cb, {"type": "segment_error", "seg": i, "error": str(e)})
        
        # Merge all segments
        try:
            # Create video metadata (would be passed from server)
            video_metadata = {
                "duration_s": 0.0,  # Would be calculated
                "width": 1920,
                "height": 1080,
                "fps": 30.0
            }
            
            # Merge segments
            merged_tracks = merge_all(job_id, segment_results, video_metadata)
            
            # Generate player insights
            merger = SegmentMerger(cfg)
            player_insights = merger.generate_player_insights(job_id, merged_tracks, [])
            
            # Save tracks.json
            tracks_path = merger.save_tracks_json(job_id, merged_tracks, f"/tmp/jobs/{job_id}")
            
            # Save player insights
            insights_paths = merger.save_player_insights(player_insights, f"/tmp/jobs/{job_id}")
            
            # Upload to Supabase
            key = f"jobs/{job_id}/tracks.json"
            with open(tracks_path, 'r') as f:
                tracks_data = json.load(f)
            upload_bytes(ANALYSES_BUCKET, key, json.dumps(tracks_data).encode("utf-8"), "application/json")
            tracks_url = sign_storage_path(ANALYSES_BUCKET, key, expires=86400)
            
            # Upload player insights
            for insights_path in insights_paths:
                player_id = Path(insights_path).stem.split('_')[1]
                insights_key = f"jobs/{job_id}/players/tid_{player_id}.json"
                with open(insights_path, 'r') as f:
                    insights_data = json.load(f)
                upload_bytes(ANALYSES_BUCKET, insights_key, json.dumps(insights_data).encode("utf-8"), "application/json")
            
            post_cb(cb, {
                "type": "done",
                "total_segments": len(segs),
                "processed": processed,
                "tracks_url": tracks_url,
                "auto_tune": {}
            })
            
            return {"ok": True, "mode": "segments", "processed": processed, "tracks_url": tracks_url}
            
        except Exception as e:
            post_cb(cb, {"type": "error", "message": f"Merge failed: {str(e)}"})
            return {"ok": False, "error": str(e)}

    # ---------- Mode C: TRAIN (dataset + epochs + config) ----------
    mode = (event.get("input") or {}).get("mode")
    if mode == "train":
        inp = (event.get("input") or {})
        job_id = inp.get("job_id") or str(uuid.uuid4())[:8]
        cb = inp.get("callback_url")  # optional
        dataset = inp.get("dataset")  # e.g. 'public/datasets/my_dataset' or signed URL / R2 path / local mount
        epochs = int(inp.get("epochs", 20))
        batch = int(inp.get("batch", 16))
        imgsz = int(inp.get("imgsz", 1280))
        model = inp.get("model", "yolov8x.pt")
        project = inp.get("project", f"/tmp/train/{job_id}")
        run_name = inp.get("run_name", f"train_{job_id}")
        resume = bool(inp.get("resume", False))

        # Optional pseudo-label pass controls
        do_pseudolabels = bool(inp.get("pseudolabels", True))
        pseudolabel_source = inp.get("pseudolabel_source")  # videos folder or URL list

        # record job start in Supabase (best-effort)
        try:
            upsert_rest(
                "analysis_jobs",
                {"id": job_id, "status": "running", "kind": "train", "player_id": None, "video_url": None},
                on_conflict="id"
            )
        except Exception as e:
            print("WARN: analysis_jobs upsert running (train) failed:", e, flush=True)

        def cb_post(payload):
            if cb:
                post_cb(cb, {"job_id": job_id, **payload})

        try:
            os.makedirs(project, exist_ok=True)

            # 0) (Optional) PSEUDO-LABEL GENERATION
            if do_pseudolabels and pseudolabel_source:
                try:
                    from training.generate_pseudolabels import generate_main as _gen_pl
                    cb_post({"type": "pseudolabels_start", "source": str(pseudolabel_source)})
                    pl_out = Path(project) / "pseudolabels"
                    pl_out.mkdir(parents=True, exist_ok=True)
                    _gen_pl(
                        source=pseudolabel_source,
                        out_dir=str(pl_out),
                        model=model,
                        imgsz=imgsz,
                        conf=inp.get("pl_conf", 0.25),
                        iou=inp.get("pl_iou", 0.5),
                        device=inp.get("device", "cpu"),
                        progress_cb=lambda p: cb_post({"type": "pseudolabels_progress", **p})
                    )
                    # If dataset wasn't provided but pseudo labels were created, use them as dataset
                    if not dataset:
                        dataset = str(pl_out)
                    cb_post({"type": "pseudolabels_done", "out": str(pl_out)})
                except Exception as e:
                    cb_post({"type": "pseudolabels_error", "error": str(e)})

            if not dataset:
                raise RuntimeError("TRAIN mode requires 'dataset' (folder or YAML).")

            # 1) TRAIN
            from training.train import train_main as _train
            cb_post({
                "type": "train_start",
                "dataset": str(dataset),
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "model": model,
                "project": project,
                "run_name": run_name,
                "resume": resume
            })

            train_artifacts = _train(
                dataset=dataset,
                model=model,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                project=project,
                name=run_name,
                resume=resume,
                device=inp.get("device", "cpu"),
                progress_cb=lambda p: cb_post({"type": "train_progress", **p})
            )

            # 2) UPLOAD ARTIFACTS TO SUPABASE (best-effort)
            # Expect train_artifacts to contain keys like: weights_path, results_dir, curves_png, last_ckpt, best_ckpt
            signed = {}
            try:
                artifacts_to_push = []
                for k in ("best_ckpt", "last_ckpt", "curves_png", "results_dir"):
                    v = train_artifacts.get(k)
                    if v:
                        artifacts_to_push.append((k, v))
                for label, p in artifacts_to_push:
                    p = Path(p)
                    if p.is_file():
                        key = f"training/{job_id}/{p.name}"
                        upload_bytes(ANALYSES_BUCKET, key, p.read_bytes(), "application/octet-stream")
                        signed[label] = sign_storage_path(ANALYSES_BUCKET, key, expires=7*24*3600)
                    elif p.is_dir():
                        # push selected files from dir
                        for f in p.glob("*"):
                            if f.is_file():
                                key = f"training/{job_id}/{f.name}"
                                upload_bytes(ANALYSES_BUCKET, key, f.read_bytes(), "application/octet-stream")
                        # sign index file if exists
                        idx = p / "results.csv"
                        if idx.exists():
                            key = f"training/{job_id}/results.csv"
                            signed["results_csv"] = sign_storage_path(ANALYSES_BUCKET, key, expires=7*24*3600)
                cb_post({"type": "artifacts_uploaded", "signed": signed})
            except Exception as e:
                cb_post({"type": "artifacts_upload_error", "error": str(e)})

            # 3) FINISH
            try:
                upsert_rest(
                    "analysis_jobs",
                    {"id": job_id, "status": "done", "kind": "train", "result_url": json.dumps(signed)},
                    on_conflict="id"
                )
            except Exception as e:
                print("WARN: upsert done (train) failed:", e, flush=True)

            cb_post({"type": "done", "job_id": job_id, "signed": signed})
            return {"ok": True, "mode": "train", "job_id": job_id, "artifacts": signed}

        except Exception as e:
            tb = traceback.format_exc()
            cb_post({"type": "error", "error": str(e)})
            try:
                upsert_rest(
                    "analysis_jobs",
                    {"id": job_id, "status": "error", "kind": "train", "result_url": None},
                    on_conflict="id"
                )
            except:
                pass
            print("ERROR (train):", tb, flush=True)
            return {"ok": False, "mode": "train", "job_id": job_id, "error": str(e), "traceback": tb}

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
