import os, json, glob, time, shutil, sys, subprocess
from collections import defaultdict
from math import sqrt

RUNS="runs/json"; MODELS="ai/models"
os.makedirs(MODELS, exist_ok=True)

def run(cmd):
    print(">>", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

def best_val(path):
    if not os.path.exists(path): return None
    try:
        h = json.load(open(path)).get("history", [])
        return min((e["val_mse"] for e in h), default=None)
    except Exception:
        return None

ts=time.strftime("%Y%m%d-%H%M%S")
work=f"{RUNS}/nightly_{ts}"
os.makedirs(work, exist_ok=True)

# 1) choose latest teamfix or flat tracks
cands = sorted(glob.glob(f"{RUNS}/full_tracks_*teamfix.json") or glob.glob(f"{RUNS}/full_tracks_*flat.json"))
if not cands: sys.exit("No tracks JSON found under runs/json/")
src = cands[-1]
print("Using tracks:", src)

# 2) events
run(f"python events_from_tracks.py --in {src} --out {work}/events.json")

# 3) rewards (inline)
ev=json.load(open(f"{work}/events.json"))
last_t=0
if ev.get("possessions"): last_t=max(p["t_end"] for p in ev["possessions"])
elif ev.get("touches"):   last_t=max(t["t"] for t in ev["touches"])
elif ev.get("passes"):    last_t=max(p["t"] for p in ev["passes"])
T=last_t+1
W={'pos_frame':0.01,'pass':1.0,'touch':0.2,'turnover':-2.0}
r=[0.0]*T; r0=[0.0]*T; r1=[0.0]*T
for p in ev.get("possessions", []):
    team=p.get("team")
    if team is None: continue
    t0=max(0,p["t_start"]); t1=min(T-1,p["t_end"])
    for t in range(t0, t1+1):
        (r0 if team==0 else r1)[t]+=W['pos_frame']
for e in ev.get("touches", []):
    t=e["t"]
    if 0<=t<T: r[t]+=W["touch"]
for e in ev.get("passes", []):
    t=e["t"]
    if 0<=t<T: r[t]+=W["pass"]
for i in range(1,len(ev.get("possessions", []))):
    a,b=ev["possessions"][i-1], ev["possessions"][i]
    if a.get("team") is not None and b.get("team") is not None and a["team"]!=b["team"]:
        t=max(0,min(T-1,b["t_start"])); r[t]+=W["turnover"]
json.dump({"T":T,"reward":r,"team0_channel":r0,"team1_channel":r1,"weights":W}, open(f"{work}/reward.json","w"))
print("wrote", f"{work}/reward.json")

# 4) frame dataset (inline)
tracks=json.load(open(src))
by_t=defaultdict(list)
for rr in tracks: by_t[rr["t"]].append(rr)

def nearest(fr):
    ps=[r for r in fr if r.get("cls")==0]; bs=[r for r in fr if r.get("cls")==1]
    if not bs or not ps: return None
    bx,by=bs[0]["x"],bs[0]["y"]; hid=None; d0=1e9; team=None
    for p in ps:
        dx=p["x"]-bx; dy=p["y"]-by; d=sqrt(dx*dx+dy*dy)
        if d<d0: d0=d; hid=p["id"]; team=p.get("team")
    return bx,by,hid,(None if team is None else int(team))

import csv
csv_path=f"{work}/frame_dataset.csv"
with open(csv_path,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["t","ball_x","ball_y","holder_id","holder_team","reward","team0_r","team1_r"])
    for t in sorted(by_t):
        fr=by_t[t]
        if not any(r.get("cls")==1 for r in fr): continue
        nh=nearest(fr)
        if nh is None: continue
        bx,by,hid,ht=nh
        rwd= r[t]  if 0<=t<T else 0.0
        r0t= r0[t] if 0<=t<T else 0.0
        r1t= r1[t] if 0<=t<T else 0.0
        w.writerow([t,bx,by,hid,ht,rwd,r0t,r1t])
print("wrote", csv_path)

# 5) train + gate
report=f"{work}/training_sup_v1.json"
model=f"{work}/model_sup_v1.pt"
run(f"python ai/scoutiqo1_train_supervised.py --csv {csv_path} --epochs 6 --emb 16 --lr 1e-3 --bs 2048 --min_samples 20 --out {model} --report {report}")

new_val = best_val(report)
cur_report = f"{MODELS}/current_training.json"
cur_model  = f"{MODELS}/current.pt"
cur_val = best_val(cur_report) if os.path.exists(cur_report) else None

def promote():
    shutil.copyfile(model, cur_model)
    shutil.copyfile(report, cur_report)
    print("PROMOTED ->", cur_model)

print("new_val:", new_val, "cur_val:", cur_val)
if new_val is None: print("No val metric; keeping current.")
elif cur_val is None or new_val < cur_val: promote()
else: print("Kept current.")
