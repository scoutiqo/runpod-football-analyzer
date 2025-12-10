import json
from collections import defaultdict
from pathlib import Path
import math

IN_PATH = Path("runs/json/tracks_meters_tracked_short_merged.json")
OUT_PATH = Path("runs/json/tracks_meters_tracked_short_merged_speed.json")

FPS = 25.0  # your test_short.mp4 fps

def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Input file not found: {IN_PATH}")

    data = json.loads(IN_PATH.read_text())

    # Handle {"video": ...,"pitch": ...,"tracks":[...]} or plain list
    if isinstance(data, dict):
        tracks = data.get("tracks", [])
        header = {k: v for k, v in data.items() if k != "tracks"}
    elif isinstance(data, list):
        tracks = data
        header = None
    else:
        raise SystemExit(f"Unsupported JSON structure in {IN_PATH}")

    print(f"[add_speed] loaded {len(tracks)} rows from {IN_PATH}")

    # Group by canonical id
    by_id = defaultdict(list)
    for row in tracks:
        if not isinstance(row, dict):
            continue
        if row.get("type") != "player":
            continue
        cid = row.get("id")
        if cid is None:
            continue
        by_id[int(cid)].append(row)

    # Sort each id’s rows by time
    for cid, rows in by_id.items():
        rows.sort(key=lambda r: float(r.get("t", 0.0)))

    # Compute speeds
    n_speed = 0
    for cid, rows in by_id.items():
        prev = None
        for r in rows:
            t = float(r.get("t", 0.0))
            x = float(r.get("x_m", 0.0))
            y = float(r.get("y_m", 0.0))
            if prev is None:
                r["speed_ms"] = 0.0
                prev = (t, x, y)
                continue
            pt, px, py = prev
            dt = max(t - pt, 1.0 / FPS)  # avoid zero
            dx = x - px
            dy = y - py
            dist = math.hypot(dx, dy)
            r["speed_ms"] = dist / dt
            prev = (t, x, y)
            n_speed += 1

    print(f"[add_speed] wrote speed_ms for {n_speed} samples")

    # Write back in the same top-level shape
    if header is not None:
        out = dict(header)
        out["tracks"] = tracks
    else:
        out = tracks

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False))
    print(f"[add_speed] saved → {OUT_PATH}")

if __name__ == "__main__":
    main()
