#!/usr/bin/env python
"""
frame_player_counts_merged.py

Read merged tracks (meters) and report how many canonical player IDs
are present per frame. We assume:
  - file: runs/json/tracks_meters_tracked_short_merged.json
  - each row has fields: t (sec), type, id_canon, x_m, y_m
"""

import json
from collections import defaultdict
from pathlib import Path
import numpy as np

IN_PATH = Path("runs/json/tracks_meters_tracked_short_merged.json")


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Input file not found: {IN_PATH}")

    data = json.loads(IN_PATH.read_text())
    if isinstance(data, dict):
        tracks = data.get("tracks") or data.get("data") or data.get("items") or []
    else:
        tracks = data

    print(f"[frame_counts] Loaded {len(tracks)} rows from {IN_PATH}")

    # group by frame index inferred from t * fps (25)
    fps = 25.0
    frame_to_ids = defaultdict(set)

    for r in tracks:
        if r.get("type") != "player":
            continue
        t = float(r.get("t", 0.0))
        f = int(round(t * fps))
        cid = r.get("id_canon", r.get("id"))
        frame_to_ids[f].add(cid)

    if not frame_to_ids:
        print("[frame_counts] No player rows found.")
        return

    frames = sorted(frame_to_ids.keys())
    counts = [len(frame_to_ids[f]) for f in frames]

    print(f"Total frames with at least one player: {len(frames)}")
    print(f"Min players in a frame: {min(counts)}")
    print(f"Max players in a frame: {max(counts)}")
    print(f"Mean players per frame: {np.mean(counts):.2f}")

    print("\nTop 20 frames by player count:")
    idx_sorted = sorted(range(len(frames)), key=lambda i: counts[i], reverse=True)[:20]
    print("frame   players")
    print("-----   -------")
    for i in idx_sorted:
        print(f"{frames[i]:5d}   {counts[i]:7d}")


if __name__ == "__main__":
    main()
