# autotune.py
import argparse, json, os, random, time, requests
from datetime import datetime, timezone
def utcnow(): return datetime.now(timezone.utc).isoformat()
def sample_params():
    import random
    return {"conf_player":round(random.uniform(0.25,0.55),2),
            "conf_ball":round(random.uniform(0.10,0.25),2),
            "nms_iou":round(random.uniform(0.45,0.65),2),
            "min_track_len":random.choice([5,7,8,10,12]),
            "smooth_sigma":random.choice([0,2,3,4]),
            "kalman":random.choice([True,False])}
def run_job(endpoint, api_key, signed_url, params, seg_s):
    payload={"run_id":f"tune_{int(time.time())}_{random.randint(1000,9999)}",
      "video":{"source":"supabase","url":signed_url,"clip_strategy":{"type":"segment_if_needed","target_bitrate_kbps":3000,"max_segment_seconds":int(seg_s),"reencode_codec":"h264"}},
      "models":{"yolo_players":"yolov8x.pt","yolo_ball":"yolov8n.pt","tracker_players":"bytetrack","tracker_ball":"deepsort"},
      "params":params,"calibration":{"method":"homography_auto","pitch":"fifa_105x68"},
      "output":{"bucket":"analyses","prefix":f"tuning/{datetime.now(timezone.utc).strftime('%Y-%m-%d')}","artifacts":["tracks.json"]}}
    r=requests.post(f"https://api.runpod.ai/v2/{endpoint}/run", headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}, json=payload, timeout=120)
    r.raise_for_status(); return r.json()
def fetch_json(url): r=requests.get(url,timeout=60); r.raise_for_status(); return r.json()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--endpoint",required=True); ap.add_argument("--signed_url",required=True)
    ap.add_argument("--api_key",default=os.getenv("RUNPOD_API_KEY")); ap.add_argument("--rounds",type=int,default=8)
    ap.add_argument("--segment_seconds",type=int,default=45); ap.add_argument("--out",required=True)
    a=ap.parse_args(); assert a.api_key, "Set RUNPOD_API_KEY or pass --api_key"
    from evaluator import compute_metrics
    trials=[]; import random
    for i in range(a.rounds):
        params=sample_params(); job=run_job(a.endpoint,a.api_key,a.signed_url,params,a.segment_seconds)
        tracks_url = (job.get("artifacts") or {}).get("tracks.json") or (job.get("output") or {}).get("tracks_url")
        if not tracks_url: trials.append({"params":params,"error":"no tracks"}); continue
        tj=fetch_json(tracks_url); m=compute_metrics(tj)
        trials.append({"params":params,"score":m["stability_score"],"metrics":m,"tracks_url":tracks_url})
        print(f"[{i+1}/{a.rounds}] {m['stability_score']=} {params=}")
        time.sleep(0.4)
    best=max((t for t in trials if "score" in t), key=lambda x:x["score"], default=None)
    json.dump({"created_at":utcnow(),"best":best,"trials":trials}, open(a.out,"w",encoding="utf-8"), indent=2)
    print(json.dumps(best or {"error":"no successful trials"}, indent=2))
if __name__=="__main__": main()
