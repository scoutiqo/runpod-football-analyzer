"""
export_for_studio.py

Convert full_tracks_v4_teamfix.json (flat per-detection rows)
into the unified studio format:

- runs/json/studio_export/meta.json
- runs/json/studio_export/tracks.json
- runs/json/studio_export/events.json (empty for now)

You can adjust FPS / image size / pitch size if needed.
"""

import json
from collections import defaultdict
from pathlib import Path

# ---------- CONFIG: adjust if needed ----------
INPUT_TRACKS = Path("runs/json/full_tracks_v4_teamfix.json")
OUTPUT_DIR = Path("runs/json/studio_export")

# If you know your real values, change here:
FPS = 25.0
IMAGE_WIDTH = 1920.0
IMAGE_HEIGHT = 1080.0
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

JOB_ID = "local-test-job"  # later this will be the real ml_jobs.id
# ------------------------------------------------


def team_index_to_label(team_idx):
    """Map numeric team index to label used in UI."""
    if team_idx == 1:
        return "home"
    if team_idx == 2:
        return "away"
    return "home"  # fallback; you can change to "unknown" if you prefer


def load_rows():
    """
    Load tracking rows from INPUT_TRACKS in a tolerant way.

    Supports:
    - top-level list: [ {...}, {...} ]
    - dict with common keys: rows / frames / data / tracks
    - dict with ANY list value (fallback)
    """
    with INPUT_TRACKS.open("r") as f:
        data = json.load(f)

    # Case 1: already a list of rows
    if isinstance(data, list):
        return data

    # Case 2: dict with known list keys
    if isinstance(data, dict):
        for key in ("rows", "frames", "data", "tracks", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value

        # Case 3: fallback – pick the first list-valued field
        for key, value in data.items():
            if isinstance(value, list):
                # You can print which key we picked if you want:
                # print(f"Using list under key '{key}' from {INPUT_TRACKS}")
                return value

    # If we get here, we really don't know how to interpret it
    raise ValueError(
        f"Unexpected top-level structure in {INPUT_TRACKS}; "
        f"no list found at any key"
    )


def build_meta(frame_count: int) -> dict:
    # simple homography placeholder: linear scale, refined later
    homography = [
        [PITCH_LENGTH_M / IMAGE_WIDTH, 0.0, 0.0],
        [0.0, PITCH_WIDTH_M / IMAGE_HEIGHT, 0.0],
        [0.0, 0.0, 1.0],
    ]

    return {
        "version": 1,
        "job_id": JOB_ID,
        "fps": FPS,
        "frame_count": frame_count,
        "image_width": int(IMAGE_WIDTH),
        "image_height": int(IMAGE_HEIGHT),
        "pitch": {
            "length_m": PITCH_LENGTH_M,
            "width_m": PITCH_WIDTH_M,
            "orientation": "left-to-right",
        },
        "homography": homography,
        "camera": {
            "type": "broadcast",
        },
        "extra": {},
    }


def build_tracks(rows):
    """
    Build frames list from flat rows.

    Each row is expected to have:
      - 't' : frame index or timestamp (may be float)
      - 'id': object id
      - 'team': team label
      - 'x_m', 'y_m': coordinates in meters (if present)
      - plus any additional fields (conf, etc.)
    """
    if not rows:
        return []

    # Normalize frame indices: treat r["t"] as frame index and cast to int
    for r in rows:
        # Keep original t as seconds if you want, but ensure we also have an int frame index
        if isinstance(r.get("t"), (int, float)):
            r["_frame"] = int(r["t"])
        elif "frame" in r:
            r["_frame"] = int(r["frame"])
        else:
            # fallback: if no t/frame, just skip row
            r["_frame"] = None

    valid_rows = [r for r in rows if r["_frame"] is not None]
    if not valid_rows:
        return []

    frame_indices = [r["_frame"] for r in valid_rows]
    min_f = min(frame_indices)
    max_f = max(frame_indices)

    frames = []
    for frame_idx in range(min_f, max_f + 1):
        # rows for this frame
        frame_rows = [r for r in valid_rows if r["_frame"] == frame_idx]
        if not frame_rows:
            continue

        # use the first row's t as seconds if present
        t_sec = None
        if isinstance(frame_rows[0].get("t"), (int, float)):
            t_sec = float(frame_rows[0]["t"])

        objects = []
        ball = None

        for r in frame_rows:
            obj = {
                "id": r.get("id"),
                "team": r.get("team", "unknown"),
                "x_m": r.get("x_m"),
                "y_m": r.get("y_m"),
                "conf": r.get("conf"),
            }

            # crude ball detection if you have cls or type
            if r.get("cls") == "ball" or r.get("is_ball"):
                ball = {
                    "visible": True,
                    "x_m": r.get("x_m"),
                    "y_m": r.get("y_m"),
                    "conf": r.get("conf"),
                }
            else:
                objects.append(obj)

        frames.append(
            {
                "f": frame_idx,
                "t": t_sec if t_sec is not None else frame_idx,
                "objects": objects,
                "ball": ball,
            }
        )

    return {"frames": frames}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading rows from {INPUT_TRACKS} ...")
    rows = load_rows()
    print(f"Loaded {len(rows)} rows")

    tracks = build_tracks(rows)
    frame_count = len(tracks["frames"])
    print(f"Built {frame_count} frames")

    meta = build_meta(frame_count)

    # For now, no events wired → empty list
    events = []

    (OUTPUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    (OUTPUT_DIR / "tracks.json").write_text(json.dumps(tracks, indent=2))
    (OUTPUT_DIR / "events.json").write_text(json.dumps(events, indent=2))

    print(f"Wrote studio export to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
