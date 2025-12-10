#!/usr/bin/env python
"""
build_player_team_index_short.py

Inputs:
  - runs/json/tracks_meters_tracked_short_merged.json
  - runs/json/player_team_assign_short.json

Output:
  - runs/json/player_team_index_short.json  (canonical_id -> {team, index})
  - runs/json/player_team_index_short.txt   (human-readable mapping)

We take all player ids, group by team A/B, sort by n_samples descending,
and assign indices 1..N per team (cap at 11 if you want strict squad size).
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

INPUT_TRACKS = Path("runs/json/tracks_meters_tracked_short_merged.json")
INPUT_TEAM_ASSIGN = Path("runs/json/player_team_assign_short.json")

OUT_JSON = Path("runs/json/player_team_index_short.json")
OUT_TXT = Path("runs/json/player_team_index_short.txt")

MAX_PER_TEAM = 11  # cap to 11 players per team label


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
    if not INPUT_TRACKS.exists():
        raise SystemExit(f"Missing tracks: {INPUT_TRACKS}")
    if not INPUT_TEAM_ASSIGN.exists():
        raise SystemExit(f"Missing team assignment: {INPUT_TEAM_ASSIGN}")

    tracks = load_tracks(INPUT_TRACKS)
    print(f"[team_index] Loaded {len(tracks)} rows from {INPUT_TRACKS}")

    assign_obj = json.loads(INPUT_TEAM_ASSIGN.read_text())
    assign = assign_obj.get("assign", {})

    # Count samples per canonical id
    n_by_id: Dict[int, int] = defaultdict(int)
    for r in tracks:
        if r.get("type") != "player":
            continue
        cid = int(r.get("id"))
        n_by_id[cid] += 1

    # Group ids by team label
    ids_by_team: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for cid, n in n_by_id.items():
        team = assign.get(str(cid)) or assign.get(cid) or "other"
        ids_by_team[team].append((cid, n))

    # sort by n_samples desc
    for team in ids_by_team:
        ids_by_team[team].sort(key=lambda x: (-x[1], x[0]))

    index_map: Dict[int, Dict[str, object]] = {}

    for team_label in ["A", "B"]:
        arr = ids_by_team.get(team_label, [])
        for idx, (cid, n) in enumerate(arr[:MAX_PER_TEAM], start=1):
            index_map[cid] = {"team": team_label, "index": idx, "n_samples": n}

    # all others -> team 'other', index None
    for team_label, arr in ids_by_team.items():
        for cid, n in arr:
            if cid in index_map:
                continue
            index_map[cid] = {"team": team_label, "index": None, "n_samples": n}

    OUT_JSON.write_text(json.dumps(index_map, indent=2))
    print(f"[team_index] Wrote index map JSON to {OUT_JSON}")

    # Human-readable
    with OUT_TXT.open("w") as f:
        f.write("cid   team  idx  n_samples\n")
        f.write("----  ----  ---  ---------\n")
        for team_label in ["A", "B", "other"]:
            # gather those ids
            subset = [(cid, meta) for cid, meta in index_map.items() if meta["team"] == team_label]
            subset.sort(key=lambda x: (x[1]["team"], (x[1]["index"] or 999), -x[1]["n_samples"]))
            for cid, meta in subset:
                idx = meta["index"]
                n = meta["n_samples"]
                idx_str = f"{idx:2d}" if idx is not None else " -"
                f.write(f"{cid:4d}  {team_label:>4s}  {idx_str}  {n:9d}\n")

    print(f"[team_index] Wrote text report to {OUT_TXT}")


if __name__ == "__main__":
    main()
