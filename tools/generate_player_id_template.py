#!/usr/bin/env python
"""
generate_player_id_template.py

Reads:
  runs/json/tracks_meters_tracked_short.json

Writes:
  runs/json/player_id_template_short.json

Purpose:
  Collapse hundreds of tracker IDs into a short list of main player IDs,
  with stats ready for manual/ML assignment of team + shirt number.
"""

import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

INPUT = "runs/json/tracks_meters_tracked_short.json"
OUT   = "runs/json/player_id_template_short.json"
TOP_N = 25  # number of main IDs to keep


def main():
    p = Path(INPUT)
    if not p.exists():
        raise SystemExit(f"Input file not found: {INPUT}")

    data = json.loads(p.read_text())
    tracks = data.get("tracks") or data
    if not isinstance(tracks, list):
        raise SystemExit("Unexpected format in tracks_meters_tracked_short.json")

    # group by player id
    by_id = defaultdict(list)
    for r in tracks:
        t = float(r.get("t", 0.0))
        rid = r.get("id")
        typ = r.get("type", "player")
        if typ != "player" or rid is None:
            continue
        if "x_m" in r and "y_m" in r:
            by_id[rid].append((t, float(r["x_m"]), float(r["y_m"])))

    # compute speed stats per id
    rows = []
    for pid, seq in by_id.items():
        if len(seq) < 2:
            continue
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
            "raw_id": pid,
            "n_samples": len(arr),
            "mean_v": float(arr.mean()),
            "p50_v": float(np.percentile(arr, 50)),
            "p95_v": float(np.percentile(arr, 95)),
            "max_v": float(arr.max()),
            "team": None,
            "shirt_no": None,
        }
        rows.append(row)

    # sort by n_samples descending and take top N
    rows = sorted(rows, key=lambda r: r["n_samples"], reverse=True)[:TOP_N]

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"[main-id-template] Found {len(by_id)} player ids total.")
    print(f"[main-id-template] Wrote top {len(rows)} ids to {OUT}")
    for r in rows:
        print(
            f"  raw_id={r['raw_id']:4d}  "
            f"n_samples={r['n_samples']:4d}  "
            f"mean_v={r['mean_v']:5.2f}  "
            f"p95_v={r['p95_v']:5.2f}  "
            f"max_v={r['max_v']:5.2f}"
        )


if __name__ == "__main__":
    main()
