from pathlib import Path
import json
import math
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parent
tracks_path = ROOT / "runs/json/tracks_short_pipeline.json"
model_path = ROOT / "models/pass_classifier_v1.pt"
out_path = ROOT / "runs/json/events_short_learned.json"

print(f"Tracks: {tracks_path}")
print(f"Model:  {model_path}")

# ---- load tracks ----
data = json.loads(tracks_path.read_text())
tracks = data.get("tracks", [])

players = []
ball_points = []
for tr in tracks:
    typ = str(tr.get("type", "")).lower()
    if typ == "player":
        players.append(tr)
    elif typ == "ball":
        ball_points.append(tr)

players_by_t = defaultdict(list)
for p in players:
    players_by_t[p["t"]].append(p)

ball_points_sorted = sorted(ball_points, key=lambda b: b["t"])

print(f"Player points: {len(players)}, ball points: {len(ball_points_sorted)}")
if not ball_points_sorted:
    raise SystemExit("No ball points, cannot compute passes")

# ---- load model + normalization ----
payload = torch.load(model_path, map_location="cpu")
team_map = {
    None: 0,
    "unknown": 0,
    "home": 1,
    "away": 2,
    "ref": -1
}

class PassNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

model = PassNet()
# Fix key names: map "0.weight" -> "layers.0.weight", etc.
state = {}
for k, v in payload["model_state"].items():
    new_key = f"layers.{k}"
    state[new_key] = v

model.load_state_dict(state)
model.eval()

mu = torch.tensor(payload["mu"], dtype=torch.float32)
sigma = torch.tensor(payload["sigma"], dtype=torch.float32)

def dist(a, b):
    dx = a["x_px"] - b["x_px"]
    dy = a["y_px"] - b["y_px"]
    return math.hypot(dx, dy)

touches = []
passes = []
possessions = []

current_possessor = None
current_team = None
possession_start_t = None

for i, b in enumerate(ball_points_sorted):
    t = b["t"]
    ps = players_by_t.get(t)
    if not ps:
        continue

    # nearest player to ball
    nearest = min(ps, key=lambda p: dist(p, b))
    d_near = dist(nearest, b)
    pid = nearest["id"]
    team = nearest.get("team")

    # ball movement vs previous ball sample
    if i > 0:
        prev_b = ball_points_sorted[i - 1]
        ball_move = dist(prev_b, b)
        dt = t - prev_b["t"]
    else:
        ball_move = 0.0
        dt = 0

    if current_possessor is None:
        current_possessor = pid
        current_team = team
        possession_start_t = t
        touches.append({
            "t": t,
            "player_id": pid,
            "team": team,
            "reason": "first_touch"
        })
        continue

    # same player keeps the ball
    if pid == current_possessor:
        continue

    # candidate pass event -> let the model decide
    team_code = team_map.get(team, 0)
    feat = torch.tensor([[float(t), d_near, ball_move, dt, team_code]], dtype=torch.float32)
    feat_norm = (feat - mu) / sigma
    with torch.no_grad():
        prob = float(model(feat_norm)[0, 0].item())

    if prob < 0.5:
        # model says "not a real pass"
        continue

    # close old possession
    possessions.append({
        "start_t": possession_start_t,
        "end_t": t,
        "player_id": current_possessor,
        "team": current_team,
    })

    passes.append({
        "t": t,
        "from_player_id": current_possessor,
        "to_player_id": pid,
        "from_team": current_team,
        "to_team": team,
        "prob": prob,
    })

    touches.append({
        "t": t,
        "player_id": pid,
        "team": team,
        "reason": "ml_pass",
    })

    # start new possession
    current_possessor = pid
    current_team = team
    possession_start_t = t

# close final possession if any
if current_possessor is not None and possession_start_t is not None:
    last_t = ball_points_sorted[-1]["t"]
    possessions.append({
        "start_t": possession_start_t,
        "end_t": last_t,
        "player_id": current_possessor,
        "team": current_team,
    })

out = {
    "touches": touches,
    "possessions": possessions,
    "passes": passes,
}
out_path.write_text(json.dumps(out, indent=2))
print(f"Wrote {out_path} with {len(touches)} touches, {len(possessions)} possessions, {len(passes)} passes")
