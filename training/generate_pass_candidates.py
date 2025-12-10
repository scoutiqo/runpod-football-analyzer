from pathlib import Path
import json
import math
from collections import defaultdict
import cv2

ROOT = Path(__file__).resolve().parents[1]

video_path = ROOT / "uploads/test_short.mp4"
tracks_path = ROOT / "runs/json/tracks_short_pipeline.json"
labels_path = ROOT / "training/labels/test_short_passes.json"
out_path = ROOT / "training/datasets/test_short_candidates.json"

print(f"Video:  {video_path}")
print(f"Tracks: {tracks_path}")
print(f"Labels: {labels_path}")

# --- 1) Read FPS to convert time_sec -> frame index ---
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise SystemExit(f"Could not open video {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.release()
print(f"FPS (raw): {fps}")

FRAME_SKIP = 3  # pipeline.py default
proc_fps = fps / FRAME_SKIP
print(f"Processed FPS (after frame_skip={FRAME_SKIP}): {proc_fps}")

# --- 2) Load tracks (already in processed frame index t) ---
tracks_data = json.loads(tracks_path.read_text())
tracks = tracks_data.get("tracks", [])

players = []
ball_points = []
for tr in tracks:
    typ = str(tr.get("type", "")).lower()
    if typ == "player":
        players.append(tr)
    elif typ == "ball":
        ball_points.append(tr)

print(f"Player points: {len(players)}, ball points: {len(ball_points)}")
if not ball_points:
    raise SystemExit("No ball points – cannot generate candidates")

players_by_t = defaultdict(list)
for p in players:
    players_by_t[p["t"]].append(p)

ball_points_sorted = sorted(ball_points, key=lambda b: b["t"])

def dist(a, b):
    dx = a["x_px"] - b["x_px"]
    dy = a["y_px"] - b["y_px"]
    return math.hypot(dx, dy)

# --- 3) Convert your time_sec labels -> target processed-frame indices ---
if labels_path.exists():
    raw_labels = json.loads(labels_path.read_text())
    label_ts = []
    for item in raw_labels:
        sec = float(item["time_sec"])
        t_est = int(round(sec * proc_fps))
        label_ts.append(t_est)
    print("Ground-truth pass times (frames):", label_ts)
else:
    label_ts = []
    print("WARNING: no labels file found, all candidates will be label=0")

# --- 4) Generate candidate possession-changes from tracks ---
candidates = []
current_possessor = None

for i, b in enumerate(ball_points_sorted):
    t = b["t"]
    ps = players_by_t.get(t)
    if not ps:
        continue

    nearest = min(ps, key=lambda p: dist(p, b))
    d_near = dist(nearest, b)
    pid = nearest["id"]
    team = nearest.get("team")

    # ball movement since previous ball sample
    if i > 0:
        prev_b = ball_points_sorted[i - 1]
        ball_move = dist(prev_b, b)
        dt = t - prev_b["t"]
    else:
        ball_move = 0.0
        dt = 0

    if current_possessor is None:
        current_possessor = pid
        continue

    if pid != current_possessor:
        candidates.append({
            "t": t,
            "old_id": current_possessor,
            "new_id": pid,
            "new_team": team,
            "dist_near": d_near,
            "ball_move": ball_move,
            "dt": dt,
        })
        current_possessor = pid

print(f"Generated {len(candidates)} raw candidates")

# --- 5) Attach label: 1 if candidate.t is close to any labeled frame, else 0 ---
MAX_DELTA_T = 3  # frames tolerance

for c in candidates:
    t = c["t"]
    label = 0
    for gt_t in label_ts:
        if abs(gt_t - t) <= MAX_DELTA_T:
            label = 1
            break
    c["label"] = label

num_pos = sum(1 for c in candidates if c["label"] == 1)
num_neg = len(candidates) - num_pos
print(f"Labeled candidates: {len(candidates)} (pos={num_pos}, neg={num_neg})")

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(candidates, indent=2))
print(f"Wrote dataset to {out_path}")
