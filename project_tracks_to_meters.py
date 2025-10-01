import json, argparse, os, numpy as np, math
def load(p): 
    with open(p,'r',encoding='utf-8') as f: 
        return json.load(f)

def bc(bb):
    x1,y1,x2,y2 = bb
    return ( (x1+x2)/2.0, y2 )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', required=True)                 # runs/json/teams.json
    ap.add_argument('--H', required=True)                      # runs/json/homography.json
    ap.add_argument('--out_tracks', default='runs/json/teams_metric.json')
    ap.add_argument('--out_speeds', default='runs/logs/speeds_ms.csv')
    args = ap.parse_args()

    data = load(args.tracks)
    H = np.array(load(args.H)['H'], dtype=float)              # 3x3

    frames = data.get('frames', [])

    # project player bottom-center to meters
    for fr in frames:
        for p in fr.get('players', []):
            bb = p.get('xyxy') or p.get('bbox')
            if not bb: 
                continue
            x,y = bc(bb)
            pt = np.array([x, y, 1.0], dtype=float)
            mp = H @ pt
            if mp[2] == 0: 
                continue
            mx, my = (mp[0]/mp[2], mp[1]/mp[2])
            p['meter_xy'] = [float(mx), float(my)]

    os.makedirs(os.path.dirname(args.out_tracks), exist_ok=True)
    with open(args.out_tracks, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # compute speeds (m/s) per tid
    from collections import defaultdict
    tid_pts = defaultdict(list)  # tid -> [(t, mx, my)]
    for fr in frames:
        t = fr.get('t')
        if t is None: 
            continue
        for p in fr.get('players', []):
            tid = p.get('tid')
            m = p.get('meter_xy')
            if tid is not None and m is not None:
                tid_pts[tid].append((t, m[0], m[1]))

    os.makedirs(os.path.dirname(args.out_speeds), exist_ok=True)
    with open(args.out_speeds, 'w', encoding='utf-8') as f:
        f.write('tid,count,avg_m_s,max_m_s\n')
        for tid, arr in tid_pts.items():
            arr.sort()
            v = []
            for i in range(1, len(arr)):
                dt = max(1e-6, arr[i][0]-arr[i-1][0])
                dx = arr[i][1]-arr[i-1][1]
                dy = arr[i][2]-arr[i-1][2]
                v.append(math.hypot(dx, dy)/dt)
            if v:
                f.write(f'{tid},{len(v)},{sum(v)/len(v):.2f},{max(v):.2f}\n')

    print(f"[DONE] {args.out_tracks} and {args.out_speeds}")

if __name__ == '__main__':
    main()
