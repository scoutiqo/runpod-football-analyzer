# auto_tune.py
import json, time, math, itertools, requests
from pathlib import Path
from copy import deepcopy
import config
from pipeline import run_pipeline

def score_run(out):
    """Higher is better. Unlabeled heuristic:
       - ball_coverage: fraction of frames with a ball point
       - players_per_frame: prefer 10-22
       - speed_sanity: penalize huge speed spikes (noisy tracks)
    """
    tracks = out.get("tracks", [])
    frames = set(round(r["t"],3) for r in tracks)
    n_frames = len(frames) or 1
    balls = [r for r in tracks if r.get("type")=="ball"]
    ball_cov = len({round(b["t"],3) for b in balls})/n_frames

    players = [r for r in tracks if r.get("type")=="player"]
    by_t = {}
    for r in players:
        by_t.setdefault(round(r["t"],3), 0)
        by_t[round(r["t"],3)] += 1
    ppl = [v for _,v in by_t.items()]
    ppl_avg = sum(ppl)/len(ppl) if ppl else 0.0
    ppl_pen = -abs(ppl_avg-14)/14.0  # best near ~14 visible on broadcast

    speeds = [float(r.get("speed_pxps",0)) for r in players if isinstance(r.get("speed_pxps"),(int,float))]
    bad = sum(1 for s in speeds if s>800)  # unrealistic spikes
    sp_pen = -bad/max(1,len(speeds))

    return 0.6*ball_cov + 0.3*(1+ppl_pen)/2 + 0.1*(1+sp_pen)/2, {
        "ball_coverage": round(ball_cov,3),
        "players_per_frame": round(ppl_avg,2),
        "bad_speed_frac": round(bad/max(1,len(speeds)),3)
    }

def try_configs(video, base_cfg, seconds=60, frame_skip=3):
    fps_hint = 30
    max_frames = int(seconds*fps_hint/frame_skip)

    grid = {
        "detector.conf": [0.15, 0.25, 0.35],
        "detector.iou":  [0.40, 0.50],
        "ball.min_conf":[0.05, 0.10, 0.20],
        "tracking.min_hits":[2,3,4]
    }
    keys = list(grid.keys())
    best = None

    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = deepcopy(base_cfg)
        cfg.setdefault("detector", {}).setdefault("backend","yolov8")
        cfg.setdefault("tracking",{"max_age":30,"min_hits":3})
        cfg.setdefault("ball",{"min_conf":0.10,"class_id":32})

        for k,v in zip(keys, vals):
            top, sub = k.split(".")
            cfg.setdefault(top,{})[sub] = v

        t0 = time.time()
        out = run_pipeline(video, cfg=cfg, max_frames=max_frames, frame_skip=frame_skip)
        sc, diag = score_run(out)
        took = time.time()-t0
        print(f"[{took:5.1f}s] score={sc:.3f}  diag={diag}  cfg={cfg['detector']|cfg['tracking']|cfg['ball']}")

        if not best or sc > best[0]:
            best = (sc, cfg, diag)

    return best

def upsert_config(cfg, config_id="default"):
    url = f"{config.SUPABASE_URL}/rest/v1/pipeline_config?on_conflict=id"
    h = {"Authorization": f"Bearer {config.SERVICE_ROLE}", "apikey": config.SERVICE_ROLE, "Content-Type":"application/json"}
    payload = {"id": config_id, "config": cfg}
    r = requests.post(url, headers=h, json=payload, timeout=30)
    r.raise_for_status()
    return True

if __name__ == "__main__":
    base = config.fetch_config("default")
    # force known weights so ball class 32 exists
    base.setdefault("detector",{})
    base["detector"]["weights_bucket"] = "local"
    base["detector"]["weights_path"] = "yolov8n.pt"

    video = "test_match.mp4"  # change if needed
    best = try_configs(video, base, seconds=90, frame_skip=5)
    score, best_cfg, diag = best
    print("\nBEST SCORE:", score, "DIAG:", diag)
    upsert_config(best_cfg)
    print("Updated Supabase pipeline_config (id=default).")
