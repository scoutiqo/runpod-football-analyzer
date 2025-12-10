import json, argparse, math
def dist(a,b): return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
def iter_frames(tracks):
    by_t={}
    for r in tracks: by_t.setdefault(r["t"],[]).append(r)
    for t in sorted(by_t): yield t, by_t[t]
def nearest(players, ball_xy, max_r=110):
    best=(None,1e9)
    for p in players:
        d=dist((p["x"],p["y"]), ball_xy)
        if d<best[1]: best=(p["id"], d)
    return best if best[1]<=max_r else (None,best[1])
def is_player(r):
    return r.get('cls') in (0, 'person', 'player')

def is_ball(r):
    return r.get('cls') in (1, 'ball')

def build_events(tracks):
    touches, possessions, passes=[],[],[]
    cur_team=None; cur_start=None; last_holder=None
    for t, fr in iter_frames(tracks):
        players=[r for r in fr if is_player(r)]
        balls=[r for r in fr if is_ball(r)]
        if not balls: continue
        bx,by=balls[0]["x"], balls[0]["y"]
        holder,_=nearest(players,(bx,by),max_r=110)
        if holder is not None:
            holder_team=next((r.get("team") for r in players if r["id"]==holder), None)
            if last_holder!=holder:
                touches.append({"t":t,"player_id":holder,"team":holder_team,"type":"touch"})
                if last_holder is not None:
                    passes.append({"t":t,"from":last_holder,"to":holder})
                last_holder=holder
            if cur_team is None:
                cur_team,cur_start=holder_team,t
            elif holder_team!=cur_team:
                possessions.append({"t_start":cur_start,"t_end":t-1,"team":cur_team})
                cur_team,cur_start=holder_team,t
    if cur_team is not None and cur_start is not None:
        last_t=max(r["t"] for r in tracks)
        possessions.append({"t_start":cur_start,"t_end":last_t,"team":cur_team})
    passnet={}
    for p in passes:
        k=(p["from"],p["to"]); passnet[k]=passnet.get(k,0)+1
    passnet_list=[{"from":a,"to":b,"count":c} for (a,b),c in passnet.items()]
    return {"touches":touches,"possessions":possessions,"passes":passes,"passnet":passnet_list}
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",dest="inp",required=True)
    ap.add_argument("--out",dest="out",required=True)
    args=ap.parse_args()
    data=json.load(open(args.inp))
    tracks = data if isinstance(data, list) else (data.get("tracks") or [])
    out=build_events(tracks)
    json.dump(out, open(args.out,"w"), indent=2)
    print("Wrote", args.out, "with", len(out["touches"]),"touches,",len(out["possessions"]),"possessions,",len(out["passes"]), "passes")
