import json
from pathlib import Path

LS_PATH = Path("labelstudio_exports/project-1-at-2025-11-17-14-04-4ecda89a.json")
TRACKS_PATH = Path("runs/json/tracks_wrapped_short_pb_ball.json")  # short clip with players+ball fields

def load_events():
    data = json.loads(LS_PATH.read_text(encoding="utf-8"))
    events = []
    for task in data:
        annotations = task.get("annotations", [])
        if not annotations:
            continue
        results = annotations[0].get("result", [])
        for r in results:
            if r.get("type") != "timelinelabels":
                continue
            value = r.get("value", {})
            ranges = value.get("ranges") or []
            if not ranges:
                continue
            start = ranges[0].get("start")
            try:
                frame = int(start)
            except (TypeError, ValueError):
                continue
            labels = value.get("timelinelabels") or []
            if not labels:
                continue
            label = labels[0]
            events.append({"frame": frame, "label": label})
    events.sort(key=lambda e: e["frame"])
    return events

def load_tracks():
    data = json.loads(TRACKS_PATH.read_text(encoding="utf-8"))
    fps = data.get("fps") or data.get("meta", {}).get("fps") or 25
    frames = {}
    raw_frames = data.get("frames") or []
    for fr in raw_frames:
        idx = fr.get("frame_index") or fr.get("frame") or fr.get("idx")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        frames[idx] = fr  # store full frame dict
    return fps, frames

def main():
    print(f"Loading events from {LS_PATH} ...")
    events = load_events()
    print(f"Loaded {len(events)} events")

    print(f"Loading tracks from {TRACKS_PATH} ...")
    fps, frames = load_tracks()
    print(f"Tracks: fps={fps}, frames={len(frames)}")

    if not events:
        print("⚠️  No events parsed.")
        return

    print("\nFirst up to 20 events with context:\n")
    for ev in events[:20]:
        frame = ev["frame"]
        label = ev["label"]
        fr = frames.get(frame, {}) or {}
        players = fr.get("players") or []
        ball = fr.get("ball")

        n_players = len(players)
        ball_present = 1 if ball else 0

        t = frame / fps if fps else 0.0

        print(
            f"Frame {frame:5d} (t={t:6.2f}s)  "
            f"label={label:15s}  players={n_players:2d}  ball_present={ball_present}"
        )

    print("\nDone.")

if __name__ == "__main__":
    main()
