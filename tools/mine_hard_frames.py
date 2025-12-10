import json, cv2, os, sys, numpy as np

TRACKS = sys.argv[1] if len(sys.argv)>1 else "runs/json/full_tracks_v4_teamfix.denoised.json"
VIDEO  = sys.argv[2] if len(sys.argv)>2 else "viewer/media/source.mp4"
OUTDIR = sys.argv[3] if len(sys.argv)>3 else "mined_frames"

MIN_CONF = 0.55   # low conf
IOU_THR  = 0.5    # overlaps
STEP     = 12     # sample every N frames to limit volume
MAX_PER_BUCKET = 400

def iou(a,b):
    ax1,ay1,aw,ah = a["x"]-a["w"]/2, a["y"]-a["h"]/2, a["w"], a["h"]
    bx1,by1,bw,bh = b["x"]-b["w"]/2, b["y"]-b["h"]/2, b["w"], b["h"]
    ax2,ay2 = ax1+aw, ay1+ah; bx2,by2 = bx1+bw, by1+bh
    ix1,iy1 = max(ax1,bx1), max(ay1,by1); ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    if inter<=0: return 0.0
    return inter / (aw*ah + bw*bh - inter)

os.makedirs(OUTDIR, exist_ok=True)
with open(TRACKS,'r') as f: data=json.load(f)

by_t={}
for r in data: by_t.setdefault(r["t"],[]).append(r)

cap=cv2.VideoCapture(VIDEO)
kept=set(); low_conf=0; overlaps=0; jitter=0

last_xy_by_id={}
def jitter_score(recs):
    s=0.0; n=0
    for r in recs:
        rid=r.get("id")
        if rid in last_xy_by_id:
            lx,ly=last_xy_by_id[rid]
            dx=r["x"]-lx; dy=r["y"]-ly
            s+=abs(dx)+abs(dy); n+=1
        last_xy_by_id[rid]=(r["x"],r["y"])
    return (s/n) if n else 0.0

bucket=0
for t in sorted(by_t.keys()):
    if t%STEP: continue
    recs = by_t[t]
    lc   = any(float(r.get("conf",1.0))<MIN_CONF for r in recs)
    ov   = False
    for i in range(len(recs)):
        for j in range(i+1,len(recs)):
            if iou(recs[i],recs[j])>IOU_THR: ov=True; break
        if ov: break
    js = jitter_score(recs)
    bad = lc or ov or js>25.0
    if not bad: continue

    cap.set(cv2.CAP_PROP_POS_FRAMES,t)
    ok,frame=cap.read()
    if not ok: continue
    fn=os.path.join(OUTDIR,f"t{t:06d}.jpg")
    cv2.imwrite(fn,frame)
    kept.add(t)
    if len(kept)>=MAX_PER_BUCKET: break

print(f"mined {len(kept)} frames into {OUTDIR}")
