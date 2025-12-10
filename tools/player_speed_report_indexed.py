#!/usr/bin/env python
"""
player_speed_report_indexed.py  (v2)

Uses:
  - runs/json/tracks_meters_tracked_short_merged.json
  - runs/json/player_team_index_short.json

Computes speeds on the fly from (x_m, y_m, t) and prints a speed report
with labels like "A_1", "B_7", etc.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

TRACKS_PATH = Path("runs/json/tracks_meters_tracked_short_merged.json")
INDEX_PATH = Path("runs/json/player_team_index_short.json")

MAX_V_KEEP = 12.0  # m/s cap for speed stats
MAX_DT = 0.6       # ignore jumps where time gap > 0.6s (to avoid crazy teleport speeds)
MIN_SAMPLES = 5    # minimum speed samples for reporting


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
    if not TRACKS_PATH.exists():
        raise SystemExit(f"Missing tracks: {TRACKS_PATH}")
    if not INDEX_PATH.exists():
        raise SystemExit(f"Missing index map: {INDEX_PATH}")

    tracks = load_tracks(TRACKS_PATH)
    index_map: Dict[str, Dict] = json.loads(INDEX_PATH.read_text())

    # group positions per canonical id
    by_id: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)  # cid -> list of (t, x_m, y_m)

    for r in tracks:
        if r.get("type") != "player":
            continue
        cid = int(r.get("id"))
        t = float(r.get("t", 0.0))
        x_m = r.get("x_m")
        y_m = r.get("y_m")
        if x_m is None or y_m is None:
            continue
        by_id[cid].append((t, float(x_m), float(y_m)))

    speeds_by_id: Dict[int, List[float]] = defaultdict(list)

    for cid, pts in by_id.items():
        if len(pts) < 2:
            continue
        # sort by time
        pts.sort(key=lambda x: x[0])
        for (t0, x0, y0), (t1, x1, y1) in zip(pts[:-1], pts[1:]):
            dt = t1 - t0
            if dt <= 0.0 or dt > MAX_DT:
                continue
            dx = x1 - x0
            dy = y1 - y0
            dist = (dx * dx + dy * dy) ** 0.5  # meters
            v = dist / dt                      # m/s
            if 0.0 < v <= MAX_V_KEEP:
                speeds_by_id[cid].append(v)

    entries: List[Tuple[str, int, float, float, float, float]] = []
    # label, n_samp, mean, p50, p95, max

    for cid, vs in speeds_by_id.items():
        if len(vs) < MIN_SAMPLES:
            continue
        vs_arr = np.array(vs, dtype=float)
        n = len(vs_arr)
        mean_v = float(np.mean(vs_arr))
        p50 = float(np.percentile(vs_arr, 50))
        p95 = float(np.percentile(vs_arr, 95))
        vmax = float(np.max(vs_arr))

        meta = index_map.get(str(cid)) or index_map.get(cid) or {}
        team = meta.get("team", "other")
        idx = meta.get("index")

        if team in ("A", "B") and idx is not None:
            label = f"{team}_{idx}"
        else:
            label = f"X_{cid}"

        entries.append((label, n, mean_v, p50, p95, vmax))

    # sort: team A, then B, then X; within team: numeric index
    def sort_key(e):
        label = e[0]
        if label.startswith("A_"):
            group = 0
            sub = int(label.split("_")[1])
        elif label.startswith("B_"):
            group = 1
            sub = int(label.split("_")[1])
        else:
            group = 2
            sub = 999
        return (group, sub)

    entries.sort(key=sort_key)

    print("label  n_samp   mean_v   p50_v   p95_v   max_v")
    print("-----  ------   ------   ------   ------  ------")
    for label, n, mean_v, p50, p95, vmax in entries:
        print(f"{label:5s}  {n:6d}   {mean_v:6.2f}   {p50:6.2f}   {p95:6.2f}  {vmax:6.2f}")


if __name__ == "__main__":
    main()
