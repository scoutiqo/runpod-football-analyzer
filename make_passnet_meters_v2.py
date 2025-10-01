import json, argparse, os, math
from collections import Counter
import numpy as np

def load(p):
    with open(p,'r',encoding='utf-8') as f: return json.load(f)

def proj_xy(H, x, y):
    pt = np.array([x,y,1.0], float)
    q = H @ pt
    if q[2] == 0: return None
    return float(q[0]/q[2]), float(q[1]/q[2])

def bb_center(bb):
    x1,y1,x2,y2 = bb
    return ( (x1+x2)/2.0, (y1+y2)/2.0 )

def dist(a,b): 
    return math.hypot(a[0]-b[0], a[1]-b[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teams_metric', required=True)          # has players with meter_xy + team
    ap.add_argument('--H', required=True)                     # runs/json/homography.json
    ap.add_argument('--out_json', default='runs/json/passes.json')
    ap.add_argument('--out_csv',  default='runs/logs/passnet.csv')
    ap.add_argument('--max_m', type=float, default=1.8, help='holder radius (meters)')
    ap.add_argument('--hold_min_frames', type=int, default=3, help='hysteresis (frames)')
    args = ap.parse_args()

    data = load(args.teams_metric)
    H = np.array(load(args.H)['H'], dtype=float)

    frames = data.get('frames', [])
    passes = []
    counts = Counter()

    last_confirmed = None   # (tid, team)
    pending = None          # candidate holder
    streak = 0

    for fr in frames:
        t = fr.get('t')
        ball_bb = fr.get('ball')
        if not ball_bb: 
            # keep hysteresis but no event
            continue

        # project ball center to meters
        cx, cy = bb_center(ball_bb)
        bm = proj_xy(H, cx, cy)
        if bm is None:
            continue

        # find nearest player in meters
        best = None
        best_d = 1e9
        for p in fr.get('players', []):
            m = p.get('meter_xy')
            if not m: 
                continue
            d = dist(bm, (m[0], m[1]))
            if d < best_d:
                best_d = d
                best = (p.get('tid'), p.get('team'))

        if best is None or best_d > args.max_m:
            # no clear holder this frame
            pending = None; streak = 0
            continue

        # hysteresis on holder change
        if pending is None or pending != best:
            pending = best
            streak = 1
        else:
            streak += 1

        if streak >= args.hold_min_frames:
            if last_confirmed is not None:
                (ptid, pteam) = last_confirmed
                (ntid, nteam) = pending
                if ntid != ptid and nteam is not None and pteam == nteam:
                    passes.append({'t': t, 'from_tid': ptid, 'to_tid': ntid, 'team': nteam})
                    counts[(ptid, ntid)] += 1
            last_confirmed = pending

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json,'w',encoding='utf-8') as f:
        json.dump({'passes': passes}, f, indent=2)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv,'w',encoding='utf-8') as f:
        f.write('from_tid,to_tid,count\n')
        for (u,v),c in sorted(counts.items(), key=lambda kv: -kv[1]):
            f.write(f'{u},{v},{c}\n')

    print(f'[DONE] saved {args.out_json} with {len(passes)} passes')
    print(f'[DONE] saved {args.out_csv} with {len(counts)} edges')
if __name__=='__main__':
    main()
