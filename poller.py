import os, time, subprocess, shlex, traceback
from pathlib import Path
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "./runs/artifacts"))
ANALYZER_CMD = os.environ.get(
    "ANALYZER_CMD",
    'python3 run_soccer.py --input "{VIDEO}" --export_json --out_json "{OUT_JSON}"'
)
BUCKET_BASE = "https://dirsscpuzqrjftawgotz.supabase.co/storage/v1/object/public/raw-videos/"
TABLES = [t.strip() for t in os.environ.get("JOBS_TABLES","analysis_jobs,jobs").split(",") if t.strip()]
STATUS_FIELDS = ["status","state"]
ID_FIELDS = ["id","job_id","pk"]
FILE_FIELDS = ["filename","raw_key","file_path","path","file","url","video_url","storage_key"]

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing Supabase credentials"); raise SystemExit(1)

def sb() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def _safe_update(client: Client, table: str, row: dict, fields: dict):
    # Only update columns that exist in the row
    payload = {k:v for k,v in fields.items() if k in row or k in STATUS_FIELDS or k in ("pct","progress","tracks_url","render_url","error")}
    id_val = None
    for k in ID_FIELDS:
        if k in row:
            id_val = row[k]; break
    if id_val is None:
        print("⚠️ No id field to update in table", table); return
    try:
        client.table(table).update(payload).eq(k, id_val).execute()
    except Exception as e:
        print("⚠️ Update failed:", e)

def _download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️ Downloading {url}")
    subprocess.check_call(["curl","-L","-o",str(out_path),url])
    if not out_path.exists():
        raise RuntimeError("Download failed")

def _run(video: Path, out_json: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = shlex.split(ANALYZER_CMD.format(VIDEO=str(video), OUT_JSON=str(out_json)))
    subprocess.check_call(cmd)

def _first_table_with_queued(client: Client):
    for table in TABLES:
        try:
            res = client.table(table).select("*").order("created_at", desc=False).limit(5).execute()
        except Exception as e:
            print(f"ℹ️ Skip {table}: {e}"); continue
        rows = getattr(res,"data",[]) or []
        if not rows: 
            continue
        # Find a row with status/state == queued (case-insensitive)
        for r in rows:
            status_val = None
            for s in STATUS_FIELDS:
                if s in r:
                    status_val = r[s]; break
            if (isinstance(status_val,str) and status_val.lower()=="queued"):
                return table, r
    return None, None

def _file_field(row: dict):
    for f in FILE_FIELDS:
        if f in row and row[f]:
            return f, row[f]
    return None, None

def _id_field(row: dict):
    for f in ID_FIELDS:
        if f in row:
            return f, row[f]
    return None, None

def main():
    client = sb()
    print("🟢 Worker started. Watching for queued jobs...")
    while True:
        table, job = _first_table_with_queued(client)
        if not job:
            time.sleep(3); continue

        id_key, job_id = _id_field(job)
        file_key, file_val = _file_field(job)
        status_key = next((s for s in STATUS_FIELDS if s in job), STATUS_FIELDS[0])

        print(f"🎯 Picked job from {table}: {id_key}={job_id}, file_field={file_key}")

        # Mark running (only columns that exist will be updated)
        _safe_update(client, table, job, {status_key:"running","pct":1})

        try:
            # Build URL
            if not file_val:
                raise RuntimeError("No filename/raw_key/url field found on job")

            if str(file_val).startswith("http"):
                video_url = str(file_val)
            else:
                video_url = BUCKET_BASE + str(file_val).lstrip("/")

            print(f"🎬 Using video URL: {video_url}")
            job_dir = ARTIFACT_DIR / str(job_id)
            video_path = job_dir / "source.mp4"
            out_json = job_dir / "tracks.json"

            _safe_update(client, table, job, {"pct":10})
            _download(video_url, video_path)

            _safe_update(client, table, job, {"pct":60})
            _run(video_path, out_json)

            _safe_update(client, table, job, {status_key:"done","pct":100,"tracks_url":str(out_json),"render_url":str(video_path)})
            print(f"✅ Job {job_id} completed.")

        except Exception as e:
            _safe_update(client, table, job, {status_key:"error","pct":0,"error":f"{type(e).__name__}: {e}"})
            print("❌ Error:", e); print(traceback.format_exc())
            time.sleep(3)

if __name__ == "__main__":
    main()
