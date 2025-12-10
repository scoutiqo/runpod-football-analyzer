import json, csv, sys, math
from collections import defaultdict

events = "runs/json/events_v4.json"                   # existing events
tracks = "runs/json/full_tracks_v4_teamfix.json"      # has team field
outcsv = "runs/json/pass_success_v1.csv"

ev = json.load(open(events))
tr = json.load(open(tracks))
by_t = defaultdict(list)
for r in tr: by_t[r["t"]].append(r)

def holder_team_at(t, hid):
    for r in by_t.get(t, []):
        if r.get("cls")==0 and r["id"]==hid:
            return r.get("team")
    return None

def ball_xy_at(t):
    for r in by_t.get(t, []):
        if r.get("cls")==1: return r["x"], r["y"]
    return None, None

# Simple success rule: a pass at time t is "successful" if within next 20 frames
# the ball holder stays on the same team as passer's team (no turnover window).
H=20
rows=[]
for p in ev.get("passes", []):
    t = int(p["t"])
    a = p.get("from"); b = p.get("to")
    if a is None or b is None: continue
    team_a = holder_team_at(t, a)
    if team_a is None: continue
    bx,by = ball_xy_at(t)
    if bx is None: continue
    # look ahead window
    success = 0
    for dt in range(1, H+1):
        tt = t+dt
        # nearest player to ball at tt
        fr = by_t.get(tt, [])
        balls = [r for r in fr if r.get("cls")==1]
        players = [r for r in fr if r.get("cls")==0]
        if not balls or not players: continue
        bbx,bby = balls[0]["x"], balls[0]["y"]
        best=None; bd=1e9
        for r in players:
            d=(r["x"]-bbx)**2 + (r["y"]-bby)**2
            if d<bd: bd=d; best=r
        if best:
            team_h = best.get("team")
            if team_h is not None and team_h==team_a:
                success = 1
                break
            # if explicit team switch detected, early stop fail
            if team_h is not None and team_h!=team_a:
                success = 0
                break
    rows.append([t,a,b,team_a,bx,by,success])

with open(outcsv,"w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["t","from_id","to_id","team_from","ball_x","ball_y","success"])
    w.writerows(rows)

print("Wrote", outcsv, "rows:", len(rows))
