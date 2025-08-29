# ai_agent.py
import argparse, json, os, pathlib
from datetime import datetime, timezone
SAFE={"conf_player":(0.2,0.7),"conf_ball":(0.1,0.4),"nms_iou":(0.4,0.8),"min_track_len":(3,15),"smooth_sigma":(0,6)}
def clamp(v,lo,hi): return max(lo,min(hi,v))
def utcnow(): return datetime.now(timezone.utc).isoformat()
def decide(params, metrics):
    p=dict(params); score=metrics.get("stability_score",0); ball_cov=metrics.get("ball_coverage",0)
    if score<60:
        p["conf_player"]=clamp(p.get("conf_player",0.35)-0.05,*SAFE["conf_player"])
        p["smooth_sigma"]=clamp(int(p.get("smooth_sigma",3))+1,*SAFE["smooth_sigma"])
        p["min_track_len"]=clamp(int(p.get("min_track_len",8))+1,*SAFE["min_track_len"])
    if ball_cov<0.5:
        p["conf_ball"]=clamp(p.get("conf_ball",0.15)-0.03,*SAFE["conf_ball"])
    return p
def write_patch(repo_root, out_patch, changes):
    params_path=pathlib.Path(repo_root)/"params.json"; old={}
    if params_path.exists():
        try: old=json.loads(params_path.read_text(encoding="utf-8") or "{}")
        except Exception: old={}
    new=dict(old); new.update(changes)
    params_path.write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding="utf-8")
    patch=f"--- params.json (old)\n+++ params.json (new)\n@@\n- {json.dumps(old,ensure_ascii=False)}\n+ {json.dumps(new,ensure_ascii=False)}\n"
    pathlib.Path(out_patch).write_text(patch,encoding="utf-8")
    return str(params_path)
def maybe_push_config(next_params):
    url=os.getenv("SUPABASE_URL"); key=os.getenv("SUPABASE_SERVICE_ROLE")
    if not url or not key: return {"pushed":False,"reason":"missing supabase env"}
    try:
        from supabase import create_client
        supa=create_client(url,key)
        supa.table("next_run_configs").insert({"created_at":utcnow(),"params":next_params}).execute()
        return {"pushed":True}
    except Exception as e:
        return {"pushed":False,"reason":str(e)}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tracks",required=True); ap.add_argument("--repo_root",default="."); ap.add_argument("--out_patch",required=True)
    ap.add_argument("--current_params",default=None)
    a=ap.parse_args()
    from evaluator import compute_metrics
    tj=json.load(open(a.tracks,"r",encoding="utf-8"))
    metrics=compute_metrics(tj)
    cur={"conf_player":0.35,"conf_ball":0.15,"nms_iou":0.5,"min_track_len":8,"smooth_sigma":3,"kalman":True}
    if a.current_params and os.path.exists(a.current_params):
        try: cur.update(json.load(open(a.current_params,"r",encoding="utf-8")))
        except Exception: pass
    nxt=decide(cur, metrics); path=write_patch(a.repo_root, a.out_patch, nxt); pushed=maybe_push_config(nxt)
    print(json.dumps({"computed_at":utcnow(),"metrics":metrics,"current_params":cur,"next_params":nxt,"params_path":path,"supabase":pushed}, indent=2))
if __name__=="__main__": main()
