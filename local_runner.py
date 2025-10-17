import os, sys, json, requests

BASE  = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8080")
TOKEN = os.environ.get("CALLBACK_SECRET", "supersecret123")

RUNPOD_URL = os.environ.get("RUNPOD_HANDLER_URL")
RUNPOD_KEY = os.environ.get("RUNPOD_API_KEY")

def post_event(job_id, payload):
    url = f"{BASE}/progress/{job_id}"
    r = requests.post(url, json=payload, headers={"x-callback-token": TOKEN}, timeout=60)
    print(f"POST {url} {payload} -> {r.status_code} {r.text[:200]!r}")
    r.raise_for_status()

def call_runpod_sync(segment_url: str) -> dict:
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

def forward_progress_from_response(job_id, seg, resp: dict):
    out = (resp or {}).get("output") or {}
    # 1) steps: list of strings
    steps = out.get("steps") or []
    for s in steps:
        post_event(job_id, {"type":"status","seg":seg,"msg":str(s)})
    # 2) logs / log: strings or list
    logs = out.get("logs") or out.get("log") or []
    if isinstance(logs, str):
        logs = [logs]
    for line in logs:
        post_event(job_id, {"type":"status","seg":seg,"msg":str(line)})
    # 3) progress: list of dicts with msg/pct
    prog = out.get("progress") or []
    for p in prog:
        msg = p.get("msg") if isinstance(p, dict) else str(p)
        pct = p.get("pct") if isinstance(p, dict) else None
        ev  = {"type":"status","seg":seg,"msg":msg}
        if pct is not None:
            ev["pct"] = pct
        post_event(job_id, ev)
    # 4) artifacts: name->url
    arts = out.get("artifacts") or {}
    for k,v in arts.items():
        if isinstance(v, str) and v.startswith("http"):
            post_event(job_id, {"type":"artifact","seg":seg,"name":k,"url":v})

def run(job_id, segs):
    for i, u in enumerate(segs):
        post_event(job_id, {"type":"segment_start","seg":i,"url":u})
        post_event(job_id, {"type":"status","seg":i,"msg":"dispatching to RunPod"})

        try:
            resp = call_runpod_sync(u)
            forward_progress_from_response(job_id, i, resp)
            post_event(job_id, {"type":"segment_done","seg":i})
        except Exception as e:
            post_event(job_id, {"type":"error","seg":i,"msg":str(e)})
            raise
    post_event(job_id, {"type":"job_done"})

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
