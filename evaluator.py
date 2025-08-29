import json, time, os
from collections import defaultdict

def utcnow():
    import datetime as dt
    return dt.datetime.utcnow().isoformat()+"+00:00"

def _infer_duration(tracks):
    ts = [float(r.get("t", 0.0)) for r in tracks if isinstance(r, dict) and "t" in r]
    return (max(ts)-min(ts)) if ts else 0.0

def load_tracks(path):
    with open(path, "r", encoding="utf-8") as f:
        tj = json.load(f)
    if isinstance(tj, list):
        return {"video": {"duration_s": _infer_duration(tj)}, "tracks": tj}
    if isinstance(tj, dict) and "tracks" in tj:
        v = tj.get("video", {})
        if "duration_s" not in v:
            v["duration_s"] = _infer_duration(tj["tracks"])
            tj["video"] = v
        return tj
    tracks = tj.get("data", tj.get("results", tj))
    if isinstance(tracks, list):
        return {"video": {"duration_s": _infer_duration(tracks)}, "tracks": tracks}
    return {"video": {"duration_s": 0.0}, "tracks": []}

def compute_metrics(tj):
    tracks = tj.get("tracks", [])
    duration = float(tj.get("video", {}).get("duration_s", 0.0))
    by_player = defaultdict(list)
    ball_ts, all_ts = [], []

    for r in tracks:
        if not isinstance(r, dict): 
            continue
        t = r.get("t")
        if t is None:
            continue
        t = float(t)
        all_ts.append(t)
        if r.get("type") == "player":
            pid = r.get("id")
            if pid is not None:
                by_player[pid].append(r)
        elif r.get("type") == "ball":
            ball_ts.append(t)

    players_count = len(by_player)
    total_samples = max(1, len(all_ts))
    continuity_vals = [(len(arr)/total_samples) for arr in by_player.values()]
    players_continuity_avg = sum(continuity_vals)/len(continuity_vals) if continuity_vals else 0.0

    conf_vals = []
    for arr in by_player.values():
        good = 0
        for r in arr:
            if ("x_m" in r and "y_m" in r) or ("x_px" in r and "y_px" in r):
                good += 1
        conf_vals.append(good/max(1,len(arr)))
    players_confidence_avg = sum(conf_vals)/len(conf_vals) if conf_vals else 0.0

    ball_coverage = (len(ball_ts)/total_samples) if total_samples>0 else 0.0

    import numpy as np
    jitters = []
    for pid, arr in by_player.items():
        arr = sorted(arr, key=lambda z: z["t"])
        xs, ys = [], []
        for r in arr:
            if "x_m" in r and "y_m" in r:
                xs.append(float(r["x_m"])); ys.append(float(r["y_m"]))
            elif "x_px" in r and "y_px" in r:
                xs.append(float(r["x_px"])); ys.append(float(r["y_px"]))
        if len(xs) >= 3:
            xs = np.asarray(xs); ys = np.asarray(ys)
            dx = np.diff(xs); dy = np.diff(ys)
            jitter = float(np.mean(np.sqrt(dx*dx + dy*dy)))
            jitters.append(jitter)
    stability_score = 1.0/(1.0 + (float(np.mean(jitters))/100.0)) if jitters else 0.0

    return {
        "players_count": players_count,
        "players_continuity_avg": round(players_continuity_avg, 4),
        "players_confidence_avg": round(players_confidence_avg, 4),
        "ball_coverage": round(ball_coverage, 4),
        "stability_score": round(stability_score, 4),
    }

def push_to_supabase(run_id, out_json):
    try:
        from supabase import create_client, Client
    except Exception:
        return {"pushed": False, "reason": "No module named 'supabase'"}
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE")
    if not (url and key):
        return {"pushed": False, "reason": "Missing SUPABASE_URL/KEY"}
    sb = create_client(url, key)
    data = {"run_id": run_id, "payload": out_json, "created_at": utcnow()}
    try:
        sb.table("metrics").insert(data).execute()
        return {"pushed": True}
    except Exception as e:
        return {"pushed": False, "reason": str(e)}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run_id", default=None)
    args = ap.parse_args()

    tj = load_tracks(args.tracks)
    run_id = args.run_id or tj.get("run_id") or f"local_{int(time.time())}"
    m = compute_metrics(tj)
    out = {"run_id": run_id, "computed_at": utcnow(), "metrics": m}
    out["supabase"] = push_to_supabase(run_id, out)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
