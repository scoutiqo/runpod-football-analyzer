import json
from pathlib import Path
from collections import defaultdict
import math

INP = Path("runs/json/tracks_meters_tracked_short_merged.json")
OUT = Path("runs/json/tracks_meters_tracked_short_merged_speed.json")

data = json.loads(INP.read_text())
tracks = data.get("tracks", data if isinstance(data, list) else [])

# group by canonical_id
by_id = defaultdict(list)
for r in tracks:
    if r.get("type") != "player":
        continue
    cid = r.get("canonical_id")
    if cid is None:
        continue
    by_id[cid].append(r)

def dist(a, b):
    dx = b["x_m"] - a["x_m"]
    dy = b["y_m"] - a["y_m"]
    return math.sqrt(dx*dx + dy*dy)

FPS = 25.0  # we know this clip is 25 fps

# compute per-row speed
for cid, rows in by_id.items():
    rows.sort(key=lambda r: r["t"])
    prev = None
    for r in rows:
        if prev is None:
            r["v_mps"] = 0.0
        else:
            dt = r["t"] - prev["t"]
            if dt <= 0:
                r["v_mps"] = 0.0
            else:
                d = dist(prev, r)
                v = d / dt
                # clamp to something human (12 m/s ~ 43 km/h)
                if v > 12.0:
                    v = 12.0
                r["v_mps"] = v
        prev = r

OUT.write_text(json.dumps({"tracks": tracks}, ensure_ascii=False))
print(f"[add_speed] wrote {OUT} with {len(tracks)} rows")
