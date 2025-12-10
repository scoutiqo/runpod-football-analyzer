import json, sys, math

IN  = sys.argv[1] if len(sys.argv)>1 else "runs/json/full_tracks_v4_teamfix.json"
OUT = sys.argv[2] if len(sys.argv)>2 else "runs/json/full_tracks_v4_teamfix.denoised.json"

MIN_CONF   = float(sys.argv[3]) if len(sys.argv)>3 else 0.65    # raise/lower if needed
MIN_H      = int(sys.argv[4])  if len(sys.argv)>4 else 34       # drop tiny ghosts
MAX_H      = int(sys.argv[5])  if len(sys.argv)>5 else 400      # drop huge boxes
NMS_IOU    = float(sys.argv[6]) if len(sys.argv)>6 else 0.45
SMOOTH_EMA = float(sys.argv[7]) if len(sys.argv)>7 else 0.25    # 0=no smoothing, 0.25 mild

def iou(a,b):
    ax1,ay1,aw,ah = a["x"]-a["w"]/2, a["y"]-a["h"]/2, a["w"], a["h"]
    bx1,by1,bw,bh = b["x"]-b["w"]/2, b["y"]-b["h"]/2, b["w"], b["h"]
    ax2, ay2 = ax1+aw, ay1+ah
    bx2, by2 = bx1+bw, by1+bh
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih   = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter    = iw*ih
    if inter<=0: return 0.0
    area_a   = aw*ah
    area_b   = bw*bh
    return inter / (area_a + area_b - inter)

# stream-ish (load once; 500MB-ish file still OK on this box)
with open(IN, "r") as f:
    data = json.load(f)

# group by frame
by_t = {}
for r in data:
    t = r["t"]
    by_t.setdefault(t, []).append(r)

# optional smoothing state: last xyz per id
last = {}

def nms(recs):
    # sort by confidence desc (missing conf -> 0)
    recs = sorted(recs, key=lambda r: float(r.get("conf",0.0)), reverse=True)
    kept = []
    for r in recs:
        keep = True
        for k in kept:
            if iou(r,k) >= NMS_IOU:
                keep = False
                break
        if keep: kept.append(r)
    return kept

out = []
for t in sorted(by_t.keys()):
    recs = by_t[t]
    # 1) basic filters
    recs = [r for r in recs
            if float(r.get("conf",1.0)) >= MIN_CONF
            and MIN_H <= r.get("h",0) <= MAX_H]

    # 2) separate ball vs players (avoid NMS merging ball with players)
    players = [r for r in recs if r.get("cls",0)!=1]
    ball    = [r for r in recs if r.get("cls",0)==1]

    # 3) NMS on players
    players = nms(players)

    # 4) smoothing (EMA) on center xy per id *within frame sequence*
    def smooth(r):
        rid = (r.get("id"), r.get("cls",0))
        if rid in last:
            lx, ly = last[rid]
            x = (1-SMOOTH_EMA)*r["x"] + SMOOTH_EMA*lx
            y = (1-SMOOTH_EMA)*r["y"] + SMOOTH_EMA*ly
        else:
            x, y = r["x"], r["y"]
        last[rid] = (x,y)
        r = dict(r); r["x"], r["y"] = x, y
        return r

    players = [smooth(r) for r in players]
    ball    = [smooth(r) for r in ball] if ball else ball

    out.extend(players + ball)

with open(OUT, "w") as f:
    json.dump(out, f)

print(f"wrote {OUT}  frames={len(by_t)}  out_objs={len(out)}")
