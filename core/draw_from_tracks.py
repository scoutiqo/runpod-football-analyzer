import os
import json
import cv2

TRACKS_PATH = "runs/json/tracks_players_ball_simple.json"
VIDEO_IN = "uploads/test_short.mp4"          # change if your source video name is different
VIDEO_OUT = "runs/videos/test_short_annotated.mp4"

os.makedirs("runs/videos", exist_ok=True)

print("Loading tracks from:", TRACKS_PATH)
with open(TRACKS_PATH, "r") as f:
    data = json.load(f)

frames = data.get("frames", [])
print("Frames in JSON:", len(frames))

# Build quick index: frame_idx -> list of objects
frame_index = {}
for fr in frames:
    frame_id = fr.get("frame")
    objs = fr.get("objects", [])
    frame_index[frame_id] = objs

print("Indexed frames:", len(frame_index))

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video {VIDEO_IN}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    objs = frame_index.get(frame_idx, [])

    for obj in objs:
        cls_id = obj.get("cls", 0)   # 0 = player, 1 = ball (once we have it)
        tid = obj.get("id", -1)

        x1 = int(obj["x1"])
        y1 = int(obj["y1"])
        x2 = int(obj["x2"])
        y2 = int(obj["y2"])

        if cls_id == 0:
            # player box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"p-{tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        elif cls_id == 1:
            # ball box (will show once we actually detect it)
            cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 8, (0, 255, 255), 2)
            cv2.putText(frame, f"ball-{tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("Wrote", VIDEO_OUT)
