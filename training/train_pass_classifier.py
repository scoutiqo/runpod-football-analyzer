import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
dataset_path = ROOT / "training/datasets/test_short_candidates.json"
out_model = ROOT / "models/pass_classifier_v1.pt"

print(f"Loading dataset: {dataset_path}")

data = json.loads(dataset_path.read_text())

# Features we will use
# ---------------------
# t ................. frame index
# dist_near ......... player-ball distance
# ball_move ......... how much the ball traveled since last sample
# dt ................ time between ball samples
# new_team .......... we map to int: home=1, away=2, unknown=0, ref=-1
#
# label ............. 1 = real pass, 0 = not a pass

team_map = {
    None: 0,
    "unknown": 0,
    "home": 1,
    "away": 2,
    "ref": -1
}

X = []
y = []

for c in data:
    feat = [
        c["t"],
        c["dist_near"],
        c["ball_move"],
        c["dt"],
        team_map.get(c["new_team"], 0)
    ]
    X.append(feat)
    y.append(c["label"])

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

print("Dataset size:", len(X))

# Normalize X:
mu = X.mean(dim=0, keepdim=True)
sigma = X.std(dim=0, keepdim=True) + 1e-6
Xn = (X - mu) / sigma

# Very small MLP:
model = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=0.001)

epochs = 30

for epoch in range(epochs):
    opt.zero_grad()
    ypred = model(Xn)
    loss = loss_fn(ypred, y)
    loss.backward()
    opt.step()

    if epoch % 5 == 0 or epoch == epochs-1:
        with torch.no_grad():
            preds = (ypred > 0.5).float()
            acc = (preds == y).float().mean().item()
        print(f"Epoch {epoch:02d} | Loss={loss.item():.4f} | Acc={acc:.3f}")

# Save everything: model + normalization stats
payload = {
    "model_state": model.state_dict(),
    "mu": mu.tolist(),
    "sigma": sigma.tolist()
}

torch.save(payload, out_model)
print(f"Saved trained model to {out_model}")
