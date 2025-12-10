#!/usr/bin/env python
"""
Convert runs/json/tracks_players_ball_simple.json
into runs/json/tracks_raw_short.json (flat list with x_px/y_px for calibrate.py).
"""

import json
from pathlib import Path

FPS = 25.0
SRC = Path("runs/json/tracks_players_ball_simple.json")
DST = Path("runs/json/tracks_raw_short.json")


def main():
    if not SRC.exists():
        raise SystemExit(f"Source file not found: {SRC}")

    data = json.loads(SRC.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    tracks = []

    for fr in frames:
        fno = int(fr.get("frame", 0))
        t = fno / FPS
        objs = fr.get("objects", [])
        for o in objs:
            cls = o.get("cls")
            # map YOLO class to type
            if cls == 0:
                typ = "player"
            elif cls == 32:
                typ = "ball"
            else:
                # ignore other classes (e.g. tv, ads)
                continue

            x1 = float(o.get("x1", 0.0))
            y1 = float(o.get("y1", 0.0))
            x2 = float(o.get("x2", 0.0))
            y2 = float(o.get("y2", 0.0))

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            tracks.append(
                {
                    "t": t,
                    "type": typ,
                    "id": int(o.get("id", -1)),
                    "x_px": cx,
                    "y_px": cy,
                    "cls": int(cls) if cls is not None else None,
                    "conf": float(o.get("conf", 0.0)),
                }
            )

    out = {"tracks": tracks}
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[make_tracks_raw_short] wrote {DST} with {len(tracks)} items")


if __name__ == "__main__":
    main()
