#!/usr/bin/env python
"""
player_speed_report.py

Input:
  runs/json/tracks_meters_tracked_short.json

Output:
  Per-player speed summary table:
    id, n_samples, mean_v, p50_v, p95_v, max_v
"""

import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np


INPUT = "runs/json/tracks_meters_tracked_short.json"


def main():
    p = Path(INPUT)
    if not p.exists():
        raise SystemExit(f"Input file not found: {INPUT}")

    data = json.loads(p.read_text())
    tracks = data.get("tracks") or data
    if not isinstance(tracks, list):
        raise SystemExit("Unexpected format in tracks_meters_tracked_short.json")

    # group tracks by (type, id)
    by_id = defaultdict(list)
    for r in tracks:
        t = float(r.get("t", 0.0))
        rid = r.get("id", None)
        typ = r.get("type", "player")
        # only players for now
        if typ != "player":
            continue
        if rid is None:
            continue
        if "x_m" in r and "y_m" in r:
            by_id[rid].append((t, float(r["x_m"]), float(r["y_m"])))

    print(f"Found {len(by_id)} player ids")

    rows = []
    for pid, seq in by_id.items():
        if len(seq) < 2:
            continue
        # sort by time
        seq = sorted(seq, key=lambda x: x[0])
        speeds = []
        for (t1, x1, y1), (t2, x2, y2) in zip(seq, seq[1:]):
            dt = t2 - t1
            if dt <= 0:
                continue
            dx = x2 - x1
            dy = y2 - y1
            v = math.hypot(dx, dy) / dt
            speeds.append(v)

        if not speeds:
            continue

        arr = np.array(speeds)
        row = {
            "id": pid,
            "n_samples": len(arr),
            "mean_v": float(arr.mean()),
            "p50_v": float(np.percentile(arr, 50)),
            "p95_v": float(np.percentile(arr, 95)),
            "max_v": float(arr.max()),
        }
        rows.append(row)

    # sort: most samples first
    rows = sorted(rows, key=lambda r: r["n_samples"], reverse=True)

    print("\nPer-player speed stats:")
    print("id    n_samp   mean_v   p50_v   p95_v   max_v")
    print("----  -------  -------  -------  -------  -------")
    for r in rows:
        print(f"{r['id']:4d}  {r['n_samples']:7d} "
              f"{r['mean_v']:7.2f} {r['p50_v']:7.2f} {r['p95_v']:7.2f} {r['max_v']:7.2f}")


if __name__ == "__main__":
    main()
