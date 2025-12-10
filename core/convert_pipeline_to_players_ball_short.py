from pathlib import Path
import json

SRC = Path("runs/json/tracks_short_pipeline.json")
DST = Path("runs/json/tracks_wrapped_short_pb.json")

data = json.loads(SRC.read_text())
frames = data.get("frames") or data.get("by_frame") or data.get("data") or []
if isinstance(frames, dict):
    frames = list(frames.values())

for fr in frames:
    objs = fr.get("objects", fr.get("detections", []))
    players, ball = [], None

    for o in objs:
        cls = o.get("cls", o.get("class", o.get("category_id")))
        label = (o.get("label") or "").lower()
        if label in ("player", "person") or cls == 0:
            players.append(o)
        if label in ("ball", "football") or cls == 1:
            ball = o

    fr["players"] = players
    fr["ball"] = ball

out = {"frames": frames}
DST.write_text(json.dumps(out))
print(f"Wrote {DST} with {len(frames)} frames")
