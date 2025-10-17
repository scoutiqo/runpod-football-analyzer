import os, sys, json, time, requests

BASE  = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8080")
TOKEN = os.environ.get("CALLBACK_SECRET", "supersecret123")

RP_KEY = os.environ.get("RUNPOD_API_KEY")           # rp_...
RP_EID = os.environ.get("RUNPOD_ENDPOINT_ID")       # e.g. br8xmojkyvacw7
RP_BASE= f"https://api.runpod.ai/v2/{RP_EID}"

def post_event(job_id, payload):
    url = f"{BASE}/progress/{job_id}"
    r = requests.post(url, json=payload, headers={"x-callback-token": TOKEN}, timeout=60)
    print(f"POST {url} {payload} -> {r.status_code} {r.text[:200]!r}")
    r.raise_for_status()

def rp_headers():
    if not RP_KEY or not RP_EID:
        raise RuntimeError("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in environment.")
    return {"Authorization": f"Bearer {RP_KEY}", "Content-Type": "application/json"}

def rp_submit(segment_url: str) -> str:
    body = {"input": {"segment_url": segment_url}}
    r = requests.post(f"{RP_BASE}/run", json=body, headers=rp_headers(), timeout=60)
    print(f"RUNPOD submit -> {r.status_code}")
    r.raise_for_status()
    data = r.json()
    job_id = data.get("id") or data.get("jobId") or data.get("job_id")
    if not job_id:
        raise RuntimeError(f"Missing job id in response: {data}")
    return job_id

def rp_status(job_id: str) -> dict:
    r = requests.get(f"{RP_BASE}/status/{job_id}", headers=rp_headers(), timeout=60)
    # 404s briefly happen right after submit; tiny delay & retry
    if r.status_code == 404:
        time.sleep(1.0)
        r = requests.get(f"{RP_BASE}/status/{job_id}", headers=rp_headers(), timeout=60)
    print(f"RUNPOD status {job_id} -> {r.status_code}")
    r.raise_for_status()
    return r.json()

def stream_until_done(job_id: str, seg: int, poll=3, max_wait=60*60):
    """Poll /status, forwarding new steps/logs/progress and artifacts."""
    seen_steps = 0
    seen_logs  = 0
    seen_prog  = 0
    t0 = time.time()
    last_status = None

    while True:
        if time.time() - t0 > max_wait:
            raise TimeoutError("RunPod job exceeded max_wait")

        resp = rp_status(job_id)
        status = resp.get("status") or resp.get("state") or ""
        if status != last_status and status:
            post_event(GLOBAL_JOB, {"type":"status","seg":seg,"msg":f"RunPod: {status}"})
            last_status = status

        out = (resp or {}).get("output") or {}

        # steps (list of strings)
        steps = out.get("steps") or []
        if isinstance(steps, list):
            for s in steps[seen_steps:]:
                post_event(GLOBAL_JOB, {"type":"status","seg":seg,"msg":str(s)})
            seen_steps = len(steps)

        # logs/log (string or list)
        logs = out.get("logs") or out.get("log") or []
        if isinstance(logs, str):
            logs = [logs]
        if isinstance(logs, list):
            for line in logs[seen_logs:]:
                post_event(GLOBAL_JOB, {"type":"status","seg":seg,"msg":str(line)})
            seen_logs = len(logs)

        # progress (list of dicts or strings)
        prog = out.get("progress") or []
        if isinstance(prog, list):
            for p in prog[seen_prog:]:
                if isinstance(p, dict):
                    ev = {"type":"status","seg":seg,"msg":str(p.get("msg") or p),}
                    if "pct" in p: ev["pct"] = p["pct"]
                else:
                    ev = {"type":"status","seg":seg,"msg":str(p)}
                post_event(GLOBAL_JOB, ev)
            seen_prog = len(prog)

        # artifacts (name->url)
        arts = out.get("artifacts") or {}
        for k,v in (arts.items() if isinstance(arts, dict) else []):
            if isinstance(v, str) and v.startswith("http"):
                post_event(GLOBAL_JOB, {"type":"artifact","seg":seg,"name":k,"url":v})

        if status in ("COMPLETED","COMPLETED_WITH_ERRORS","FAILED","CANCELLED","TIMED_OUT"):
            return resp

        time.sleep(poll)

def process_segment(seg_index: int, segment_url: str):
    post_event(GLOBAL_JOB, {"type":"segment_start","seg":seg_index,"url":segment_url})
    post_event(GLOBAL_JOB, {"type":"status","seg":seg_index,"msg":"dispatching to RunPod"})
    rp_job = rp_submit(segment_url)
    post_event(GLOBAL_JOB, {"type":"status","seg":seg_index,"msg":f"RunPod job {rp_job}"})
    resp = stream_until_done(rp_job, seg_index)
    # if there are terminal artifacts not yet forwarded, forward again
    out = (resp or {}).get("output") or {}
    arts = out.get("artifacts") or {}
    for k,v in (arts.items() if isinstance(arts, dict) else []):
        if isinstance(v, str) and v.startswith("http"):
            post_event(GLOBAL_JOB, {"type":"artifact","seg":seg_index,"name":k,"url":v})
    post_event(GLOBAL_JOB, {"type":"segment_done","seg":seg_index})

def run(job_id, segs):
    global GLOBAL_JOB
    GLOBAL_JOB = job_id
    for i, u in enumerate(segs):
        try:
            process_segment(i, u)
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
