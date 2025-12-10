# tools/count_players_per_frame.py
from pathlib import Path
from collections import defaultdict
import json
import numpy as np

# Use the calibrated + merged tracks
PATH = Path("runs/json/tracks_meters_tracked_short_merged.json")

def main():
    if not PATH.exists():
        raise SystemExit(f"File not found: {PATH}")

    data = json.loads(PATH.read_text())
    tracks = data.get("tracks", data if isinstance(data, list) else [])
    if not isinstance(tracks, list):
        raise SystemExit("Unexpected JSON format: 'tracks' is not a list")

    by_frame = defaultdict(int)

    for r in tracks:
        if r.get("type") != "player":
            continue
        # In this pipeline t is frame index (0,1,2,...) from tracker
        t = int(round(float(r.get("t", 0))))
        by_frame[t] += 1

    if not by_frame:
        print("No player tracks found.")
        return

    frame_idxs = sorted(by_frame.keys())
    counts = [by_frame[f] for f in frame_idxs]

    print(f"Total frames with at least one player: {len(frame_idxs)}")
    print(f"Min players in a frame: {min(counts)}")
    print(f"Max players in a frame: {max(counts)}")
    print(f"Mean players per frame: {np.mean(counts):.2f}")

    # Show a few sample frames with highest counts
    top = sorted(by_frame.items(), key=lambda kv: kv[1], reverse=True)[:20]
    print("\nTop 20 frames by player count:")
    print("frame   players")
    print("-----   -------")
    for f, c in top:
        print(f"{f:5d}   {c:7d}")

if __name__ == "__main__":
    main()
