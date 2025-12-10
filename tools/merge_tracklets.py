#!/usr/bin/env python
"""
merge_tracklets.py (D1-ID-MERGE v0)

Conservative merging of raw tracker IDs into canonical player IDs
for the short calibrated clip.

Input:
  - runs/json/tracks_meters_tracked_short.json

Outputs:
  - runs/json/tracks_meters_tracked_short_merged.json
  - runs/json/player_id_merge_map_short.json
  - runs/json/player_id_merge_report_short.txt

Merging logic (SAFE v0):
  - Only touches type == "player"
  - Never merges IDs that overlap in time
  - Only considers merges when the time gap is small (Δt <= DT_MAX)
  - Only considers merges when spatial distance at the boundary is
    physically plausible (<= v_max * Δt + margin)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ---- Config ----

IN_PATH = Path("runs/json/tracks_meters_tracked_short.json")
OUT_PATH = Path("runs/json/tracks_meters_tracked_short_merged.json")
MAP_PATH = Path("runs/json/player_id_merge_map_short.json")
REPORT_PATH = Path("runs/json/player_id_merge_report_short.txt")

# We assume t is in seconds (as in your calibrated file)
# Max physically plausible speed ~ 9 m/s (sprints).
V_MAX = 9.0

# We only consider merges for short gaps (seconds)
DT_MAX = 0.50  # 0.5s

# Extra spatial slack (meters), to forgive minor calibration noise
DIST_MARGIN = 1.0


def _load_tracks(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[merge_tracklets] Input file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tracks" in data:
        return data
    elif isinstance(data, list):
        # Wrap into dict for consistency
        return {"video": None, "pitch": None, "tracks": data}
    else:
        raise SystemExit(f"[merge_tracklets] Unexpected JSON structure in {path}")


def _group_by_id(tracks: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_id: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in tracks:
        # Only players; ignore ball / others
        if r.get("type", "player") != "player":
            continue
        if "id" not in r or "x_m" not in r or "y_m" not in r or "t" not in r:
            continue
        try:
            rid = int(r["id"])
        except (TypeError, ValueError):
            continue
        by_id[rid].append(r)

    # sort each id's records by time
    for rid, rows in by_id.items():
        rows.sort(key=lambda rr: float(rr["t"]))
    return by_id


def _interval_stats(by_id: Dict[int, List[Dict[str, Any]]]
                    ) -> Dict[int, Dict[str, Any]]:
    """
    For each raw id, compute:
      - t_start, t_end
      - (x_start, y_start), (x_end, y_end)
      - n_samples
    """
    stats: Dict[int, Dict[str, Any]] = {}
    for rid, rows in by_id.items():
        if not rows:
            continue
        t_start = float(rows[0]["t"])
        t_end = float(rows[-1]["t"])
        x_start = float(rows[0]["x_m"])
        y_start = float(rows[0]["y_m"])
        x_end = float(rows[-1]["x_m"])
        y_end = float(rows[-1]["y_m"])
        stats[rid] = {
            "t_start": t_start,
            "t_end": t_end,
            "x_start": x_start,
            "y_start": y_start,
            "x_end": x_end,
            "y_end": y_end,
            "n_samples": len(rows),
        }
    return stats


def _intervals_overlap(a: Dict[str, float], b: Dict[str, float],
                       overlap_eps: float = 1e-3) -> bool:
    """
    Check if two [t_start, t_end] intervals overlap more than ~0 time.
    """
    s1, e1 = a["t_start"], a["t_end"]
    s2, e2 = b["t_start"], b["t_end"]
    latest_start = max(s1, s2)
    earliest_end = min(e1, e2)
    return (earliest_end - latest_start) > overlap_eps


def _boundary_gap_and_distance(a: Dict[str, float],
                               b: Dict[str, float]) -> Tuple[float, float]:
    """
    For two intervals a, b, compute (dt, dist) between their closest
    non-overlapping boundary points in time.

    If b starts after a ends:
       dt = s_b - e_a, distance between (x_end_a, y_end_a) and (x_start_b, y_start_b)
    If a starts after b ends:
       dt = s_a - e_b, distance between (x_end_b, y_end_b) and (x_start_a, y_start_a)
    """
    s1, e1 = a["t_start"], a["t_end"]
    s2, e2 = b["t_start"], b["t_end"]

    if e1 <= s2:
        dt = s2 - e1
        dx = a["x_end"] - b["x_start"]
        dy = a["y_end"] - b["y_start"]
    elif e2 <= s1:
        dt = s1 - e2
        dx = b["x_end"] - a["x_start"]
        dy = b["y_end"] - a["y_start"]
    else:
        # Overlapping or ill-ordered; caller should have checked.
        return 0.0, float("inf")

    dist = math.hypot(dx, dy)
    return dt, dist


def _build_merge_graph(stats: Dict[int, Dict[str, Any]]) -> Dict[int, List[int]]:
    """
    Build an undirected graph over raw_ids, where an edge means "these
    two IDs are safe to merge".
    """
    ids = sorted(stats.keys())
    graph: Dict[int, List[int]] = defaultdict(list)

    for i_idx, rid_i in enumerate(ids):
        st_i = stats[rid_i]
        for j_idx in range(i_idx + 1, len(ids)):
            rid_j = ids[j_idx]
            st_j = stats[rid_j]

            # Never merge IDs that overlap in time
            if _intervals_overlap(st_i, st_j):
                continue

            dt, dist = _boundary_gap_and_distance(st_i, st_j)
            if dt <= 0.0 or dt > DT_MAX:
                continue

            # Max allowed distance assuming physically plausible speed
            max_dist = V_MAX * dt + DIST_MARGIN
            if dist <= max_dist:
                graph[rid_i].append(rid_j)
                graph[rid_j].append(rid_i)

    return graph


def _connected_components(graph: Dict[int, List[int]],
                          ids: List[int]) -> List[List[int]]:
    """
    Undirected graph → list of components (each a list of raw_ids).
    """
    visited = set()
    comps: List[List[int]] = []

    for rid in ids:
        if rid in visited:
            continue
        comp = []
        dq = deque([rid])
        visited.add(rid)
        while dq:
            u = dq.popleft()
            comp.append(u)
            for v in graph.get(u, []):
                if v not in visited:
                    visited.add(v)
                    dq.append(v)
        comps.append(sorted(comp))
    return comps


def _assign_canonical_ids(comps: List[List[int]],
                          stats: Dict[int, Dict[str, Any]]) -> Dict[int, int]:
    """
    For each connected component (set of raw_ids), assign a canonical_id
    (1..K). Within a component, all raw_ids map to the same canonical_id.

    We sort components by the earliest t_start in that component, so
    canonical_id is roughly chronological.
    """
    # Sort components by earliest start time
    comp_with_t: List[Tuple[float, List[int]]] = []
    for comp in comps:
        t0 = min(stats[rid]["t_start"] for rid in comp)
        comp_with_t.append((t0, comp))
    comp_with_t.sort(key=lambda x: x[0])

    mapping: Dict[int, int] = {}
    for canon_id, (_, comp) in enumerate(comp_with_t, start=1):
        for rid in comp:
            mapping[rid] = canon_id
    return mapping


def _write_report(mapping: Dict[int, int],
                  stats: Dict[int, Dict[str, Any]],
                  out_path: Path) -> None:
    # Invert mapping: canonical_id -> list of raw_ids
    by_canon: Dict[int, List[int]] = defaultdict(list)
    for rid, cid in mapping.items():
        by_canon[cid].append(rid)

    lines: List[str] = []
    lines.append("[D1-ID-MERGE v0] Player ID merge report (short clip)\n")
    lines.append(f"Total raw player ids: {len(stats)}")
    lines.append(f"Total canonical player ids: {len(by_canon)}\n")

    lines.append("Canonical groups (canonical_id: raw_ids ... n_samples_total):\n")
    for cid in sorted(by_canon.keys()):
        raw_ids = sorted(by_canon[cid])
        n_total = sum(stats[rid]["n_samples"] for rid in raw_ids)
        span_start = min(stats[rid]["t_start"] for rid in raw_ids)
        span_end = max(stats[rid]["t_end"] for rid in raw_ids)
        lines.append(
            f"  canon_id={cid:2d} | raw_ids={raw_ids} | "
            f"n_samples={n_total} | t_span=[{span_start:.2f}, {span_end:.2f}]"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[merge_tracklets] Wrote merge report to {out_path}")


def main() -> None:
    print(f"[merge_tracklets] Loading {IN_PATH} ...")
    data = _load_tracks(IN_PATH)
    tracks = data.get("tracks", [])
    print(f"[merge_tracklets] Total tracks (all types): {len(tracks)}")

    by_id = _group_by_id(tracks)
    stats = _interval_stats(by_id)

    print(f"[merge_tracklets] Raw player ids: {len(by_id)}")

    # Build merge graph and find components
    graph = _build_merge_graph(stats)
    comps = _connected_components(graph, sorted(stats.keys()))

    mapping = _assign_canonical_ids(comps, stats)
    print(f"[merge_tracklets] Canonical player ids: {len(set(mapping.values()))}")

    # Write json mapping
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    # keys as string for JSON
    json_map = {str(rid): mapping[rid] for rid in sorted(mapping.keys())}
    MAP_PATH.write_text(json.dumps(json_map, indent=2), encoding="utf-8")
    print(f"[merge_tracklets] Wrote id map to {MAP_PATH}")

    # Write text report
    _write_report(mapping, stats, REPORT_PATH)

    # Build merged tracks
    merged_tracks: List[Dict[str, Any]] = []
    for r in tracks:
        if r.get("type", "player") == "player" and "id" in r:
            try:
                rid = int(r["id"])
            except (TypeError, ValueError):
                merged_tracks.append(dict(r))
                continue
            cid = mapping.get(rid, rid)
            rr = dict(r)
            rr["id_raw"] = rid
            rr["id"] = cid
            merged_tracks.append(rr)
        else:
            # Non-player (ball, refs, etc.) pass-through untouched
            merged_tracks.append(dict(r))

    out_data = {
        "video": data.get("video"),
        "pitch": data.get("pitch"),
        "tracks": merged_tracks,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"[merge_tracklets] Wrote merged tracks to {OUT_PATH} "
          f"({len(merged_tracks)} rows)")


if __name__ == "__main__":
    main()
