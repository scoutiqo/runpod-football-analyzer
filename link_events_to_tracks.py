import json
from pathlib import Path

LS_PATH = Path("labelstudio_exports/project-1-at-2025-11-17-14-04-4ecda89a.json")

CANDIDATES = [
    Path("runs/json/tracks_short_pipeline.json"),
    Path("runs/json/tracks_wrapped_short_pb.json"),
    Path("runs/json/tracks_wrapped_short.json"),
    Path("runs/json/tracks_team_short.json"),
    Path("runs/json/tracks_wrapped.json"),
    Path("runs/json/tracks.json"),
    Path("runs/json/full_tracks_v4_teamfix.json"),
]

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

def load_frames(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    fps = data.get("fps") or data.get("meta", {}).get("fps") or 25
    frames_dict = {}

    raw_frames = data.get("frames")
    if raw_frames is None:
        # try "data" or top-level list
        if isinstance(data, list):
            raw_frames = data
        else:
            raw_frames = data.get("data") or []

    for fr in raw_frames:
        idx = fr.get("frame_index") or fr.get("frame") or fr.get("idx")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        objs = fr.get("objects") or fr.get("detections") or []
        frames_dict[idx] = objs

    return fps, frames_dict

def main():
    print(f"Reading Label Studio export: {LS_PATH}")
    events = load_events()
    print(f"Loaded {len(events)} events")
    event_frames = [e["frame"] for e in events]
    if not events:
        print("No events parsed, aborting.")
        return

    print("\nTesting candidate tracks files:\n")
    best = None

    for cand in CANDIDATES:
        if not cand.exists():
            print(f"- {cand}: MISSING")
            continue

        try:
            fps, frames = load_frames(cand)
        except Exception as e:
            print(f"- {cand}: ERROR loading ({e})")
            continue

        if not frames:
            print(f"- {cand}: loaded, but 0 frames")
            continue

        keys = sorted(frames.keys())
        min_f, max_f = keys[0], keys[-1]
        matches = sum(1 for f in event_frames if f in frames)

        print(f"- {cand}: fps={fps}, frames={len(frames)}, "
              f"range=[{min_f},{max_f}], event_matches={matches}")

        score = matches
        if best is None or score > best["score"]:
            best = {
                "path": cand,
                "fps": fps,
                "frames": frames,
                "score": score,
            }

    if not best or best["score"] == 0:
        print("\n❌ No candidate tracks file seems to cover your event frames.")
        return

    print(f"\n✅ Best match: {best['path']} (event_matches={best['score']})")

    fps = best["fps"]
    frames = best["frames"]

    print("\nSample of your events with context from best tracks file:\n")
    for ev in events[:20]:
        frame = ev["frame"]
        label = ev["label"]
        objs = frames.get(frame, [])
        n_players = sum(1 for o in objs if not o.get("is_ball"))
        n_balls = sum(1 for o in objs if o.get("is_ball"))
        t = frame / fps if fps else 0.0
        print(f"Frame {frame:5d} (t={t:6.2f}s)  label={label:15s}  "
              f"players={n_players:2d}  balls={n_balls}")

if __name__ == "__main__":
    main()
