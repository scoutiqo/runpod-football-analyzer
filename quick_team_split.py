import json, argparse, os
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import KMeans

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p, data):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def bbox_center_x(bb):
    x1,y1,x2,y2 = bb
    return (x1+x2)/2.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="tracks.json (with frames/players/tid/xyxy)")
    ap.add_argument("--out", required=True, help="teams.json")
    ap.add_argument("--smooth_window", type=int, default=25, help="frames for per-tid majority smoothing")
    args = ap.parse_args()

    data = load_json(args.tracks)
    frames = data.get("frames", [])
    # Per-frame kmeans on x-centers -> (cluster 0/1)
    per_tid_teams = defaultdict(list)  # tid -> list of team labels over time (raw)

    # First pass: assign per-frame team by x clustering
    for fr in frames:
        players = fr.get("players", [])
        if not players:
            continue
        xs = []
        idx = []
        for i,p in enumerate(players):
            bb = p.get("xyxy") or p.get("bbox")
            if not bb:
                continue
            xs.append(bbox_center_x(bb))
            idx.append(i)
        if len(xs) < 2:
            continue
        xs_np = np.array(xs).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, n_init=4, random_state=0).fit(xs_np)
        labels = kmeans.labels_.tolist()
        # Ensure left-to-right consistency: label 0 = left cluster
        left_center = np.mean([xs[j] for j,l in enumerate(labels) if l == 0]) if any(l==0 for l in labels) else 1e9
        right_center = np.mean([xs[j] for j,l in enumerate(labels) if l == 1]) if any(l==1 for l in labels) else -1e9
        # If swapped, flip labels so 0 ~ left, 1 ~ right
        if left_center > right_center:
            labels = [1 - l for l in labels]

        for j, i_player in enumerate(idx):
            tid = players[i_player].get("tid")
            if tid is None:
                continue
            per_tid_teams[tid].append(labels[j])

    # Second pass: smooth per tid (majority over sliding window); default to global majority if short
    smoothed_tid_team = {}
    for tid, labs in per_tid_teams.items():
        if not labs:
            continue
        if len(labs) < args.smooth_window:
            smoothed_tid_team[tid] = Counter(labs).most_common(1)[0][0]
        else:
            # simple majority over windowed chunks
            w = args.smooth_window
            votes = []
            for s in range(0, len(labs), w):
                chunk = labs[s:s+w]
                votes.append(Counter(chunk).most_common(1)[0][0])
            smoothed_tid_team[tid] = Counter(votes).most_common(1)[0][0]

    # Third pass: write team into frames
    out = dict(data)  # shallow copy
    out_frames = []
    for fr in frames:
        players = fr.get("players", [])
        new_players = []
        for p in players:
            q = dict(p)
            tid = p.get("tid")
            if tid in smoothed_tid_team:
                q["team"] = int(smoothed_tid_team[tid])
            new_players.append(q)
        fr2 = dict(fr)
        fr2["players"] = new_players
        out_frames.append(fr2)
    out["frames"] = out_frames

    save_json(args.out, out)
    print(f"[DONE] wrote {args.out} with team labels for {len(smoothed_tid_team)} TIDs.")

if __name__ == "__main__":
    main()
