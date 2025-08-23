from yt_dlp import YoutubeDL
import os, json, uuid, tempfile, requests, cv2
from ultralytics import YOLO
import runpod

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_ROLE = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
ANALYSES_BUCKET = os.getenv("ANALYSES_BUCKET", "analyses")

_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO("yolov8n.pt")  # can swap to yolov10n later
    return _MODEL

def dl_tmp(url: str) -> str:
    """Download to a local temp .mp4. Supports YouTube and direct .mp4 URLs."""
    # If it's YouTube, use yt-dlp to download and merge to mp4
    if "youtube.com" in url or "youtu.be" in url:
        tmpdir = tempfile.mkdtemp()
        outtmpl = os.path.join(tmpdir, "video.%(ext)s")  # yt-dlp will set the right extension
        ydl_opts = {
            "format": "mp4/bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": outtmpl,
            "quiet": True,
            "noprogress": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # find the produced mp4
        for fname in os.listdir(tmpdir):
            if fname.lower().endswith(".mp4"):
                return os.path.join(tmpdir, fname)
        raise RuntimeError("yt-dlp did not produce an mp4 file.")

    # Otherwise: direct file download
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    fd, p = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
    with open(p, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    return p


def upload_bytes(bucket, path, data, content_type):
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    h = {"Authorization": f"Bearer {SERVICE_ROLE}", "x-upsert":"true", "Content-Type": content_type}
    r = requests.post(url, headers=h, data=data, timeout=120)
    if not r.ok: raise RuntimeError(f"upload failed {r.status_code}: {r.text}")

def post_rest(table, row):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "Authorization": f"Bearer {SERVICE_ROLE}",
        "apikey": SERVICE_ROLE,
        "Content-Type": "application/json",
        "Prefer": "return=representation"  # <— ask PostgREST to return JSON
    }
    r = requests.post(url, headers=headers, data=json.dumps(row), timeout=30)
    if not r.ok:
        raise RuntimeError(f"post {table} failed {r.status_code}: {r.text}")
    # If server still returns no body, avoid JSONDecodeError
    try:
        return r.json()
    except ValueError:
        return {"status_code": r.status_code}

def patch_rest(table, where, row):
    url = f"{SUPABASE_URL}/rest/v1/{table}?{where}"
    headers = {
        "Authorization": f"Bearer {SERVICE_ROLE}",
        "apikey": SERVICE_ROLE,
        "Content-Type": "application/json",
        "Prefer": "return=representation"  # <— ask for JSON back
    }
    r = requests.patch(url, headers=headers, data=json.dumps(row), timeout=30)
    if not r.ok:
        raise RuntimeError(f"patch {table} failed {r.status_code}: {r.text}")
    try:
        return r.json()
    except ValueError:
        return {"status_code": r.status_code}


def analyze(video_url, max_frames=150, frame_skip=5):
    p = dl_tmp(video_url)
    cap = cv2.VideoCapture(p); assert cap.isOpened(), "OpenCV failed"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    model = get_model()
    tracks = []; idx=0; proc=0
    while proc<max_frames:
        ok, frame = cap.read()
        if not ok: break
        idx += 1
        if idx % frame_skip != 0: continue
        t = round(idx / fps, 3)
        res = model(frame, verbose=False)
        for r in res:
            if r.boxes is None: continue
            for b in r.boxes:
                cls = int(b.cls[0]) if b.cls is not None else -1
                if cls not in (0, 32): continue  # 0=person, 32=ball
                x1,y1,x2,y2 = b.xyxy[0].tolist()
                cx,cy = (x1+x2)/2.0,(y1+y2)/2.0
                tracks.append({"t": t, "type": "ball" if cls==32 else "player",
                               "x_px": float(cx), "y_px": float(cy)})
        proc += 1
    cap.release(); os.remove(p)
    return {"version":1, "meta":{"fps":fps,"width":W,"height":H,
                                 "frame_skip":frame_skip,"max_frames":max_frames},
            "tracks":tracks, "events":[], "metrics":{}}

def handler(event):
    inp = event.get("input", {})
    player_id  = inp.get("player_id")
    video_url  = inp.get("video_url")
    max_frames = int(inp.get("max_frames", 150))
    frame_skip = int(inp.get("frame_skip", 5))
    if not player_id or not video_url:
        return {"error":"player_id and video_url required"}

    job_id = str(uuid.uuid4())
    post_rest("analysis_jobs", {
        "id": job_id, "player_id": player_id, "status": "running", "video_url": video_url
    })

    try:
        result = analyze(video_url, max_frames=max_frames, frame_skip=frame_skip)
        key = f"players/{player_id}/{job_id}/tracks.json"
        upload_bytes(ANALYSES_BUCKET, key, json.dumps(result).encode("utf-8"), "application/json")
        patch_rest("analysis_jobs", f"id=eq.{job_id}",
                   {"status":"done", "result_bucket": ANALYSES_BUCKET, "result_path": key})
        post_rest("player_latest_analysis", {
          "player_id": player_id, "analysis_job_id": job_id,
          "result_bucket": ANALYSES_BUCKET, "result_path": key
        })
        return {"ok": True, "job_id": job_id, "bucket": ANALYSES_BUCKET, "path": key}
    except Exception as e:
        patch_rest("analysis_jobs", f"id=eq.{job_id}", {"status":"error","error":str(e)})
        return {"ok": False, "job_id": job_id, "error": str(e)}

runpod.serverless.start({"handler": handler})
