import os, sys, json, requests
from overlay_analyzer import make_overlay_for_segment

BASE  = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8081")
TOKEN = os.environ.get("CALLBACK_SECRET", "scoutsecret123")

def post(job_id, payload):
    r = requests.post(f"{BASE}/progress/{job_id}", json=payload,
                      headers={"x-callback-token": TOKEN}, timeout=30)
    print(f"POST {payload.get('type')} -> {r.status_code}")
    r.raise_for_status()

def run(job_id, segs):
    for i, url in enumerate(segs):
        post(job_id, {"type":"segment_start","seg":i,"url":url})
        post(job_id, {"type":"status","seg":i,"msg":"analyzing (opencv)..."})
        overlay_rel = make_overlay_for_segment(url, job_id, i)      # "/files/jobs/<job>/overlay_seg_000.mp4"
        overlay_url = f"{BASE}{overlay_rel}"
        post(job_id, {"type":"artifact","seg":i,"name":f"overlay_seg_{i:03}.mp4","url":overlay_url})
        post(job_id, {"type":"segment_done","seg":i})
    post(job_id, {"type":"job_done"})

if __name__ == "__main__":
    job = sys.argv[1]
    arg = sys.argv[2]
    segs = json.load(open(arg[1:], "r", encoding="utf-8-sig")) if arg.startswith("@") else json.loads(arg)
    run(job, segs)
