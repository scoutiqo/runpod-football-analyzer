#!/usr/bin/env python
"""
assign_teams_by_position_short.py

Input:
  - runs/json/tracks_meters_tracked_short_merged.json
Output:
  - runs/json/player_team_assign_short.json   (canonical_id -> team label "A"/"B"/"other")
  - runs/json/player_team_assign_short.txt    (human-readable report)

Team split is done via 1D k-means on median x_m per canonical id (k=2),
then labeling the left cluster as "A" and right cluster as "B".
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

INPUT = Path("runs/json/tracks_meters_tracked_short_merged.json")
OUT_JSON = Path("runs/json/player_team_assign_short.json")
OUT_TXT = Path("runs/json/player_team_assign_short.txt")

MIN_SAMPLES_FOR_CLUSTER = 30  # min samples for an id to participate in clustering


def load_tracks(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        tracks = data.get("tracks", data.get("data", []))
        if isinstance(tracks, dict):
            tracks = list(tracks.values())
    else:
        tracks = data
    return tracks


def main():
    if not INPUT.exists():
        raise SystemExit(f"Input not found: {INPUT}")

    tracks = load_tracks(INPUT)
    print(f"[team_pos] Loaded {len(tracks)} rows from {INPUT}")

    # Collect x_m stats per canonical player id
    xs_by_id: Dict[int, List[float]] = defaultdict(list)

    for r in tracks:
        if r.get("type") != "player":
            continue
        cid = int(r.get("id"))
        x_m = r.get("x_m")
        if x_m is None:
            continue
        xs_by_id[cid].append(float(x_m))

    id_stats: List[Tuple[int, int, float]] = []  # (cid, n_samples, median_x)
    for cid, xs in xs_by_id.items():
        n = len(xs)
        if n == 0:
            continue
        med_x = float(np.median(xs))
        id_stats.append((cid, n, med_x))

    if not id_stats:
        raise SystemExit("[team_pos] No player x_m stats found.")

    # Filter ids that participate in clustering
    cluster_ids = [(cid, n, mx) for (cid, n, mx) in id_stats if n >= MIN_SAMPLES_FOR_CLUSTER]
    if len(cluster_ids) < 2:
        print("[team_pos] WARNING: not enough ids for clustering; all will be 'other'.")
        assign = {cid: "other" for cid, _, _ in id_stats}
        OUT_JSON.write_text(json.dumps({"assign": assign}, indent=2))
        print(f"[team_pos] Wrote trivial assignment to {OUT_JSON}")
        return

    cids = [cid for (cid, _, _) in cluster_ids]
    med_xs = np.array([mx for (_, _, mx) in cluster_ids], dtype=np.float32)

    # Initialize 1D 2-means
    c_left = np.percentile(med_xs, 25)
    c_right = np.percentile(med_xs, 75)
    centers = np.array([c_left, c_right], dtype=np.float32)

    for _ in range(15):
        dists = np.abs(med_xs[:, None] - centers[None, :])  # (N,2)
        labels = np.argmin(dists, axis=1)
        new_centers = []
        for k in range(2):
            pts = med_xs[labels == k]
            if len(pts) == 0:
                new_centers.append(centers[k])
            else:
                new_centers.append(float(np.mean(pts)))
        new_centers = np.array(new_centers, dtype=np.float32)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    # Label smaller-x cluster as A, larger-x as B
    idx_left = int(np.argmin(centers))
    idx_right = 1 - idx_left

    print(f"[team_pos] centers: {centers.tolist()} -> left_idx={idx_left}, right_idx={idx_right}")

    # Build assignment for clustered ids
    assign: Dict[int, str] = {}
    for (cid, _n, mx), lab in zip(cluster_ids, labels):
        if lab == idx_left:
            assign[cid] = "A"
        elif lab == idx_right:
            assign[cid] = "B"
        else:
            assign[cid] = "other"

    # ids that didn't meet MIN_SAMPLES_FOR_CLUSTER -> assign nearest center anyway
    for cid, n, mx in id_stats:
        if cid in assign:
            continue
        dist_left = abs(mx - centers[idx_left])
        dist_right = abs(mx - centers[idx_right])
        assign[cid] = "A" if dist_left <= dist_right else "B"

    OUT_JSON.write_text(json.dumps({"assign": assign, "centers": centers.tolist()}, indent=2))
    print(f"[team_pos] Wrote assignment JSON to {OUT_JSON}")

    # Human-readable txt report
    by_team: Dict[str, List[Tuple[int, int, float]]] = {"A": [], "B": [], "other": []}
    stats_map = {cid: (n, mx) for cid, n, mx in id_stats}
    for cid, team in assign.items():
        n, mx = stats_map.get(cid, (0, float("nan")))
        by_team.setdefault(team, []).append((cid, n, mx))

    for t in by_team.values():
        t.sort(key=lambda x: (-x[1], x[0]))  # sort by n_samples desc

    with OUT_TXT.open("w") as f:
        for team_label in ["A", "B", "other"]:
            arr = by_team.get(team_label, [])
            f.write(f"=== Team {team_label} ===\n")
            f.write("cid   n_samples   median_x_m\n")
            f.write("----  ---------   -----------\n")
            for cid, n, mx in arr:
                f.write(f"{cid:4d}  {n:9d}   {mx:9.3f}\n")
            f.write("\n")

    print(f"[team_pos] Wrote text report to {OUT_TXT}")


if __name__ == "__main__":
    main()
