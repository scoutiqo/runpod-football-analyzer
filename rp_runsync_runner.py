import os, sys, json, requests

BASE  = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8080")
TOKEN = os.environ.get("CALLBACK_SECRET", "supersecret123")

RP_KEY = os.environ.get("RUNPOD_API_KEY")
RP_EID = os.environ.get("RUNPOD_ENDPOINT_ID")
RP_URL = f"https://api.runpod.ai/v2/{RP_EID}/runsync"  # sync path on purpose

def post(job_id, payload):
    r = requests.post(f"{BASE}/progress/{job_id}", json=payload, headers={"x-callback-token": TOKEN}, timeout=60)
    print("POST", payload, "->", r.status_code, r.text[:200]); r.raise_for_status()

def run(job_id, segs):
    if not RP_KEY or not RP_EID:
        raise RuntimeError("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in env.")
    headers = {"Authorization": f"Bearer {RP_KEY}", "Content-Type":"application/json"}

    for i, url in enumerate(segs):
        post(job_id, {"type":"segment_start","seg":i,"url":url})
        post(job_id, {"type":"status","seg":i,"msg":"dispatching to RunPod (runsync)"})
        body = {"input":{"segment_url": url}}
        r = requests.post(RP_URL, json=body, headers=headers, timeout=60*30)
        print("RUNPOD runsync ->", r.status_code)
        r.raise_for_status()
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {"raw": r.text}
        out  = (data or {}).get("output") or {}

        # forward steps/logs/progress if present
        for s in out.get("steps") or []:
            post(job_id, {"type":"status","seg":i,"msg":str(s)})
        logs = out.get("logs") or out.get("log") or []
        if isinstance(logs, str): logs=[logs]
        for line in logs:
            post(job_id, {"type":"status","seg":i,"msg":str(line)})
        for p in out.get("progress") or []:
            if isinstance(p, dict):
                ev={"type":"status","seg":i,"msg":str(p.get("msg") or p)}
                if "pct" in p: ev["pct"]=p["pct"]
            else:
                ev={"type":"status","seg":i,"msg":str(p)}
            post(job_id, ev)

        # artifacts: name->url
        for k,v in (out.get("artifacts") or {}).items():
            if isinstance(v, str) and v.startswith("http"):
                post(job_id, {"type":"artifact","seg":i,"name":k,"url":v})

        post(job_id, {"type":"segment_done","seg":i})

    post(job_id, {"type":"job_done"})

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: rp_runsync_runner.py <job_id> <segments_json_or_@file>"); sys.exit(2)
    arg = sys.argv[2]
    segs = json.load(open(arg[1:], "r", encoding="utf-8-sig")) if arg.startswith("@") else json.loads(arg)
    run(sys.argv[1], segs)
