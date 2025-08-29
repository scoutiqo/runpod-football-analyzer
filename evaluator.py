# evaluator.py
import argparse, json, os, statistics, time
from datetime import datetime, timezone
def utcnow(): return datetime.now(timezone.utc).isoformat()
def load_tracks(path): return json.load(open(path,'r',encoding='utf-8'))
def continuity_rate(track):
    centers = track.get("center_xy_image", [])
    if len(centers) < 2: return 1.0
    gaps=[b[0]-a[0] for a,b in zip(centers,centers[1:])]
    if not gaps: return 1.0
    med = statistics.median(gaps); bad = sum(1 for g in gaps if g>2.5*med)
    return 1.0 - bad/max(1,len(gaps))
def avg_confidence(players):
    vals=[p.get("confidence_avg",0.0) for p in players if p.get("confidence_avg") is not None]
    return sum(vals)/max(1,len(vals))
def ball_track_coverage(ball, duration_s):
    pts=ball.get("center_xy_image", [])
    if not pts or duration_s<=0: return 0.0
    dts=[b[0]-a[0] for a,b in zip(pts,pts[1:]) if b[0]>a[0]]
    if not dts: return 0.0
    med_dt=statistics.median(dts); expected=max(1,int(duration_s/med_dt))
    return min(1.0, len(pts)/expected)
def compute_metrics(tj):
    v=tj["video"]; duration=float(v.get("duration_s",0))
    players=tj.get("players",[]); ball=tj.get("ball",{})
    m={}
    m["players_count"]=len(players)
    m["players_continuity_avg"]= (sum(continuity_rate(p) for p in players)/max(1,len(players)))
    m["players_confidence_avg"]= avg_confidence(players)
    m["ball_coverage"]= ball_track_coverage(ball, duration)
    score=0; score+=35*m["players_continuity_avg"]; score+=35*min(1.0,m["players_confidence_avg"]); score+=30*m["ball_coverage"]
    m["stability_score"]= round(score,2)
    return m
def maybe_push_supabase(run_id, metrics):
    url=os.getenv("SUPABASE_URL"); key=os.getenv("SUPABASE_SERVICE_ROLE")
    if not url or not key: return {"pushed":False,"reason":"missing supabase env"}
    try:
        from supabase import create_client
        supa=create_client(url,key)
        supa.table("run_metrics").insert({"run_id":run_id,"ts":utcnow(),"metrics":metrics}).execute()
        return {"pushed":True}
    except Exception as e:
        return {"pushed":False,"reason":str(e)}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tracks",required=True); ap.add_argument("--out",required=True); ap.add_argument("--run_id",default=None)
    args=ap.parse_args()
    tj=load_tracks(args.tracks); run_id=args.run_id or tj.get("run_id") or f"local_{int(time.time())}"
    m=compute_metrics(tj); out={"run_id":run_id,"computed_at":utcnow(),"metrics":m}
    json.dump(out, open(args.out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    out["supabase"]=maybe_push_supabase(run_id,m)
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
