from pathlib import Path
import json

SRC = Path("runs/json/tracks_wrapped_short.json")
DST = Path("runs/json/tracks_wrapped_short_pb.json")

data = json.loads(SRC.read_text())
frames = data.get("frames", [])

for fr in frames:
    objs = fr.get("objects", [])
    players = []
    ball = None

    for o in objs:
        # Try multiple ways to get the class/label
        cls = o.get("cls", o.get("class", o.get("category_id")))
        label = (o.get("label") or "").lower()

        # --- ASSUMPTION ---
        # players: cls == 0 or label 'player'/'person'
        # ball:    cls == 1 or label 'ball'/'football'
        # Adjust these if your dump shows different ids/names.
        if label in ("player", "person") or cls == 0:
            players.append(o)

        if label in ("ball", "football") or cls == 1:
            # If there are more than one, keep the last – ok for now
            ball = o

    fr["players"] = players
    fr["ball"] = ball

out = {"frames": frames}
DST.write_text(json.dumps(out))
print(f"Wrote {DST} with {len(frames)} frames")
