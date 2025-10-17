import os, sys, json, time, requests

BASE  = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8080")
TOKEN = os.environ.get("CALLBACK_SECRET", "supersecret123")

RUNPOD_URL = os.environ.get("RUNPOD_HANDLER_URL")   # e.g. https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync
RUNPOD_KEY = os.environ.get("RUNPOD_API_KEY")       # rp_xxx

def post_event(job_id, payload):
    """Send a progress event to /progress/{job_id} guarded by x-callback-token."""
    url = f"{BASE}/progress/{job_id}"
    r = requests.post(url, json=payload, headers={"x-callback-token": TOKEN}, timeout=60)
    print(f"POST {url} {payload} -> {r.status_code} {r.text[:200]!r}")
    r.raise_for_status()

def call_runpod(segment_url: str) -> dict:
    """Invoke RunPod handler and return parsed JSON."""
    if not RUNPOD_URL or not RUNPOD_KEY:
        raise RuntimeError("Set RUNPOD_HANDLER_URL and RUNPOD_API_KEY in environment.")
    body = {"input": {"segment_url": segment_url}}
    headers = {"Authorization": f"Bearer {RUNPOD_KEY}", "Content-Type": "application/json"}
    r = requests.post(RUNPOD_URL, json=body, headers=headers, timeout=60*30)
    print(f"RUNPOD {RUNPOD_URL} -> {r.status_code}")
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}

def run(job_id, segs):
    for i, u in enumerate(segs):
        post_event(job_id, {"type": "segment_start", "seg": i, "url": u})
        post_event(job_id, {"type": "status", "seg": i, "msg": "dispatching to RunPod"})

        try:
            resp = call_runpod(u)
            # Stream step logs if handler provides them
            steps = ((resp or {}).get("output") or {}).get("steps") or []
            for s in steps:
                post_event(job_id, {"type": "status", "seg": i, "msg": str(s)})

            # Forward artifact URLs if present
            artifacts = ((resp or {}).get("output") or {}).get("artifacts") or {}
            for k, v in artifacts.items():
                if isinstance(v, str) and v.startswith("http"):
                    post_event(job_id, {"type": "artifact", "seg": i, "name": k, "url": v})

            post_event(job_id, {"type": "segment_done", "seg": i})

        except Exception as e:
            post_event(job_id, {"type": "error", "seg": i, "msg": str(e)})
            raise

    post_event(job_id, {"type": "job_done"})

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: local_runner.py <job_id> <segments_json_or_@file>")
        sys.exit(2)
    arg = sys.argv[2]
    if arg.startswith("@"):
        with open(arg[1:], "r", encoding="utf-8-sig") as f:
            segs = json.load(f)
    else:
        segs = json.loads(arg)
    run(sys.argv[1], segs)
