import os
import os
import json
from ultralytics import YOLO

VIDEO = "/workspace/videos/full_match_test.mp4"
import os
WEIGHTS = os.getenv("MODEL_PATH","yolov8n.pt")
OUT_JSON = "runs/json/tracks.json"

model = YOLO(WEIGHTS)

# stream=True yields a generator of Results per frame with ByteTrack IDs
gen = model.track(
    source=VIDEO,
    tracker="bytetrack.yaml",
    imgsz=1280,
    conf=0.25,
    iou=0.6,
    persist=True,
    stream=True,   # IMPORTANT: lets us iterate frames
    device=0
)

tracks = []
frame_idx = 0
for r in gen:
    frame = {"frame": frame_idx, "items": []}
    # r.boxes may contain .id for trackers
    if r.boxes is not None:
        ids = r.boxes.id
        xyxy = r.boxes.xyxy
        cls = r.boxes.cls
        conf = r.boxes.conf
        n = len(r.boxes)
        for i in range(n):
            tid = int(ids[i].item()) if ids is not None and ids[i] is not None else -1
            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
            c = float(conf[i].item()) if conf is not None else 0.0
            cl = int(cls[i].item()) if cls is not None else -1
            frame["items"].append({
                "id": tid,
                "cls": cl,
                "conf": c,
                "bbox_xyxy": [x1, y1, x2, y2]
            })
    tracks.append(frame)
    frame_idx += 1

# write JSON
out = {"video": VIDEO, "weights": WEIGHTS, "frames": tracks}
os.makedirs("runs/json", exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(out, f)
print(f"Wrote {OUT_JSON} with {len(tracks)} frames.")
