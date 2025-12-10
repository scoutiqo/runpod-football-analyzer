#!/usr/bin/env python
"""
Convert tracked detections from runs/json/tracks.json
into a generic "raw tracks" format suitable for calibrate.py.

Input:
  runs/json/tracks.json  (list of dicts like:
    { "t": frame_idx, "id": int, "x1":..., "y1":..., "x2":..., "y2":..., "conf":..., "cls": 0 })

Output:
  runs/json/tracks_raw_tracked_short.json  (list of dicts:
    {
      "t": <time_seconds>,
      "type": "player" | "other",
      "id": int,
      "x_px": float,
      "y_px": float,
      "cls": int,
      "conf": float
    }
  )

We assume:
  - t in tracks.json is frame index
  - FPS is 25 for this short clip
"""

import json
from pathlib import Path

SRC = Path("runs/json/tracks.json")
DST = Path("runs/json/tracks_raw_tracked_short.json")
FPS = 25.0  # adjust if your test_short.mp4 has different fps


def main():
    if not SRC.exists():
        raise SystemExit(f"Source file not found: {SRC}")

    data = json.loads(SRC.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list in {SRC}, got {type(data)}")

    out = []
    for r in data:
        try:
            frame_idx = int(r.get("t", 0))
        except (TypeError, ValueError):
            continue
        t_sec = frame_idx / FPS

        x1 = float(r.get("x1", 0.0))
        y1 = float(r.get("y1", 0.0))
        x2 = float(r.get("x2", 0.0))
        y2 = float(r.get("y2", 0.0))

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        cls = int(r.get("cls", -1))
        conf = float(r.get("conf", 0.0))
        pid = int(r.get("id", -1))

        # Treat cls == 0 as "player" for now (COCO: person)
        typ = "player" if cls == 0 else "other"

        out.append(
            {
                "t": t_sec,
                "type": typ,
                "id": pid,
                "x_px": cx,
                "y_px": cy,
                "cls": cls,
                "conf": conf,
            }
        )

    DST.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[make_tracks_raw_tracked_short] wrote {DST} with {len(out)} items")


if __name__ == "__main__":
    main()
