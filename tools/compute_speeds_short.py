#!/usr/bin/env python
"""
Compute simple per-player speed stats from runs/json/tracks_meters_short.json.

For each player id:
  - sort positions by time t
  - compute segment speeds (m/s)
  - print count, mean, max speed
"""

import json
import math
from collections import defaultdict
from pathlib import Path


SRC = Path("runs/json/tracks_meters_short.json")


def main():
    if not SRC.exists():
        raise SystemExit(f"Source file not found: {SRC}")

    data = json.loads(SRC.read_text(encoding="utf-8"))
    tracks = data.get("tracks", [])
    if not tracks:
        raise SystemExit("No tracks found in tracks_meters_short.json")

    # Group by player id (only type == "player")
    by_player = defaultdict(list)
    for r in tracks:
        if r.get("type") != "player":
            continue
        if "x_m" not in r or "y_m" not in r:
            continue
        pid = int(r.get("id", -1))
        t = float(r.get("t", 0.0))
        x = float(r["x_m"])
        y = float(r["y_m"])
        by_player[pid].append((t, x, y))

    print(f"Found {len(by_player)} player ids with meter_coords.")

    stats = []
    for pid, pts in by_player.items():
        if len(pts) < 2:
            continue
        pts.sort(key=lambda p: p[0])
        speeds = []
        last_t, last_x, last_y = pts[0]
        for t, x, y in pts[1:]:
            dt = t - last_t
            if dt <= 0 or dt > 2.0:
                # skip weird jumps / resets
                last_t, last_x, last_y = t, x, y
                continue
            dx = x - last_x
            dy = y - last_y
            dist = math.hypot(dx, dy)  # meters
            v = dist / dt               # m/s
            speeds.append(v)
            last_t, last_x, last_y = t, x, y

        if not speeds:
            continue
        mean_v = sum(speeds) / len(speeds)
        max_v = max(speeds)
        stats.append((pid, len(speeds), mean_v, max_v))

    # Sort players by mean speed descending
    stats.sort(key=lambda x: x[2], reverse=True)

    print("\nPer-player speed stats (top 10 by mean speed):")
    print("id    n_samples   mean_v(m/s)   max_v(m/s)")
    print("----  ---------   -----------   ----------")
    for pid, n, mean_v, max_v in stats[:10]:
        print(f"{pid:4d}  {n:9d}   {mean_v:11.3f}   {max_v:10.3f}")


if __name__ == "__main__":
    main()
