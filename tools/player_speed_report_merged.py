#!/usr/bin/env python
"""
player_speed_report_merged.py (D1-ID-REPORT v2)

Speed report per *canonical* player id, using:
  - runs/json/tracks_meters_tracked_short_merged.json

We assume this file was produced by merge_tracklets.py and contains:
  {
    "video": ...,
    "pitch": {...},
    "tracks": [
      {
        "t": <float seconds>,
        "type": "player" | "ball" | ...,
        "id": <canonical_id>,   # integer
        "id_raw": <raw tracker id>,  # optional
        "x_m": <float>,
        "y_m": <float>,
        ...
      },
      ...
    ]
  }

Output: prints table to stdout.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

IN_PATH = Path("runs/json/tracks_meters_tracked_short_merged.json")

# For reporting we treat speeds > V_MAX_REPORT as outliers and ignore them
# in summary stats (they still indicate potential calibration/ID issues).
V_MAX_REPORT = 12.0  # m/s (~43 km/h)


def _load_tracks(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[player_speed_report_merged] Input file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tracks" in data:
        return data
    elif isinstance(data, list):
        return {"video": None, "pitch": None, "tracks": data}
    else:
        raise SystemExit(f"[player_speed_report_merged] Unexpected JSON structure in {path}")


def _group_player_points(tracks: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_id: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in tracks:
        if r.get("type", "player") != "player":
            continue
        if "id" not in r or "x_m" not in r or "y_m" not in r or "t" not in r:
            continue
        try:
            cid = int(r["id"])
        except (TypeError, ValueError):
            continue
        by_id[cid].append(r)

    # sort by time
    for cid, rows in by_id.items():
        rows.sort(key=lambda rr: float(rr["t"]))
    return by_id


def _compute_speeds_for_track(points: List[Dict[str, Any]]) -> List[float]:
    """
    Given time-ordered points for one player, compute per-step speed (m/s)
    and drop NaNs / zero-dt / insane speeds > V_MAX_REPORT.
    """
    vs: List[float] = []
    for i in range(1, len(points)):
        p0, p1 = points[i - 1], points[i]
        t0, t1 = float(p0["t"]), float(p1["t"])
        dt = t1 - t0
        if dt <= 0:
            continue
        dx = float(p1["x_m"]) - float(p0["x_m"])
        dy = float(p1["y_m"]) - float(p0["y_m"])
        dist = math.hypot(dx, dy)
        v = dist / dt
        if not math.isfinite(v):
            continue
        if v <= 0:
            continue
        if v > V_MAX_REPORT:
            # treat as outlier for stats; could log later
            continue
        vs.append(v)
    return vs


def main() -> None:
    data = _load_tracks(IN_PATH)
    tracks = data.get("tracks", [])
    print(f"[player_speed_report_merged] Total tracks (all types): {len(tracks)}")

    by_id = _group_player_points(tracks)
    print(f"[player_speed_report_merged] Canonical player ids found: {len(by_id)}")

    rows = []
    for cid, pts in by_id.items():
        vs = _compute_speeds_for_track(pts)
        if not vs:
            continue
        arr = np.array(vs, dtype=float)
        n = int(arr.size)
        mean_v = float(arr.mean())
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        vmax = float(arr.max())
        rows.append((cid, n, mean_v, p50, p95, vmax))

    # sort by n_samples descending
    rows.sort(key=lambda r: r[1], reverse=True)

    print("\nPer-canonical-player speed stats (ignoring v > {:.1f} m/s):".format(V_MAX_REPORT))
    print("id   n_samp   mean_v   p50_v   p95_v   max_v")
    print("---- -------  -------  -------  -------  -------")
    for cid, n, mean_v, p50, p95, vmax in rows:
        print(f"{cid:4d} {n:7d}  {mean_v:7.2f}  {p50:7.2f}  {p95:7.2f}  {vmax:7.2f}")


if __name__ == "__main__":
    main()
