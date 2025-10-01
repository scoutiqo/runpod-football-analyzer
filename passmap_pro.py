import json, argparse, os, statistics as stats
from collections import defaultdict, Counter
import math
import matplotlib.pyplot as plt

PITCH_W, PITCH_H = 105.0, 68.0

def load(p): return json.load(open(p,'r',encoding='utf-8'))

def parse_range(s):
    if s.lower()=='all': return (None,None)
    a,b = s.split('-'); return (float(a), float(b))

def median_xy(pts):
    xs=[x for x,_ in pts]; ys=[y for _,y in pts]
    return float(stats.median(xs)), float(stats.median(ys))

def draw_half(ax):
    ax.set_facecolor('#0b1420')
    ax.add_patch(plt.Rectangle((0,0), PITCH_W, PITCH_H/2, fill=False, ec='#7a8899', lw=1.6))
    ax.add_patch(plt.Rectangle((PITCH_W*(1-40.3/105), 0), 40.3, 16.5, fill=False, ec='#536375', lw=1.2))
    ax.add_patch(plt.Rectangle((PITCH_W*(1-18.3/105), 0), 18.3, 5.5, fill=False, ec='#536375', lw=1.0))
    ax.plot([PITCH_W/2-3.66, PITCH_W/2+3.66], [0,0], color='#536375', lw=1.0)
    ax.scatter([PITCH_W/2],[11.0], s=12, c='#536375')
    ax.add_patch(plt.Circle((PITCH_W/2,11.0), 9.15, fill=False, ec='#536375', lw=1.0))
    ax.set_xlim(0,PITCH_W); ax.set_ylim(0,PITCH_H/2); ax.invert_yaxis()
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--teams_metric', required=True)
    ap.add_argument('--passes', required=True)
    ap.add_argument('--team', type=int, choices=[0,1], required=True)
    ap.add_argument('--time', default='all')
    ap.add_argument('--min_frames', type=int, default=200)
    ap.add_argument('--min_passes', type=int, default=3)
    ap.add_argument('--top_edges', type=int, default=24)
    ap.add_argument('--out', default='runs/logs/passmap_team.png')
    args=ap.parse_args()

    tm = load(args.teams_metric)
    ps = load(args.passes).get('passes', [])
    t0,t1 = parse_range(args.time)

    # collect player positions (meters) within window
    tid_pts=defaultdict(list); seen=Counter()
    for fr in tm.get('frames', []):
        t=fr.get('t')
        if t0 is not None and (t < t0 or t > t1): continue
        for p in fr.get('players', []):
            if p.get('team') != args.team: continue
            m = p.get('meter_xy')
            if not m: continue
            tid = p.get('tid')
            tid_pts[tid].append(tuple(m))
            seen[tid]+=1

    keep={tid for tid,c in seen.items() if c>=args.min_frames}
    if not keep: raise SystemExit('No stable players; lower --min_frames or widen --time')

    nodes={tid:median_xy(pts) for tid,pts in tid_pts.items() if tid in keep}

    # edges
    edge=Counter(); deg=Counter()
    for e in ps:
        if e.get('team')!=args.team: continue
        u,v=e.get('from_tid'), e.get('to_tid')
        te=e.get('t', None)
        if u in nodes and v in nodes:
            if t0 is not None and te is not None and (te<t0 or te>t1): continue
            edge[(u,v)]+=1; deg[u]+=1; deg[v]+=1

    edge=Counter({k:w for k,w in edge.items() if w>=args.min_passes})
    if args.top_edges and len(edge)>args.top_edges:
        edge=Counter(dict(edge.most_common(args.top_edges)))

    plt.figure(figsize=(6.6,10), dpi=160)
    ax=plt.gca(); draw_half(ax)

    mx=max(edge.values()) if edge else 1
    def ew(w): return 0.8+4.5*(w/mx)
    def ea(w): return 0.35+0.45*(w/mx)

    for (u,v),w in edge.items():
        x1,y1=nodes[u]; x2,y2=nodes[v]
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
            arrowprops=dict(arrowstyle='->', lw=ew(w), color=(1,1,1,ea(w))))

    if not deg: deg=Counter({tid:1 for tid in nodes})
    mdeg=max(deg.values()) if deg else 1
    for tid,(x,y) in nodes.items():
        r=2.2
        circ=plt.Circle((x,y), r, color='#ff3c91', ec='white', lw=2)
        ax.add_patch(circ)
        ax.text(x, y, str(tid), color='white', ha='center', va='center', fontsize=12, fontweight='bold')

    ttl=f'Pass Network — Team {args.team}'
    if t0 is not None: ttl+=f'  (t={int(t0)}–{int(t1)}s)'
    ax.set_title(ttl, color='white', fontsize=14, pad=14)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, facecolor='#0b1420')
    print(f'[DONE] wrote {args.out}')
if __name__=='__main__': main()
