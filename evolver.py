# evolver.py (Self-Improving Service)
import os, json, time
from fastapi import FastAPI
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from supabase import create_client  # From requirements

app = FastAPI()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SERVICE_ROLE = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supa = create_client(SUPABASE_URL, SERVICE_ROLE)

class ScoutingTuner(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 32)  # Metrics in
        self.fc2 = nn.Linear(32, 5)  # Params out (e.g., conf_player, smooth_sigma)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def load_data():
    # Pull recent metrics/tracks from Supabase (e.g., last 100 jobs)
    res = supa.table("analysis_jobs").select("metrics").order("created_at", desc=True).limit(100).execute()
    data = res.data
    metrics = np.array([d["metrics"].values() for d in data if "metrics" in d])  # e.g., stability, coverage
    targets = np.random.rand(len(metrics), 5)  # Mock; replace with labeled params from feedback
    return metrics, targets

def train_and_update(epochs=5):
    model = ScoutingTuner()
    if os.path.exists('scouting_tuner.pth'):
        model.load_state_dict(torch.load('scouting_tuner.pth'))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    metrics, targets = load_data()
    if len(metrics) < 10:
        return {"status": "insufficient data"}

    inputs = torch.tensor(metrics, dtype=torch.float32)
    labels = torch.tensor(targets, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'scouting_tuner.pth')

    # Generate code patch (e.g., update params.json)
    avg_metrics = torch.tensor(np.mean(metrics, axis=0))
    new_params = model(avg_metrics).detach().numpy().tolist()
    with open("params.json", "w") as f:
        json.dump({"conf_player": new_params[0], "conf_ball": new_params[1], ...}, f)  # Map to keys

    # Deploy update: e.g., trigger RunPod redeploy (API call)
    # requests.post("runpod_redeploy_endpoint", json={"update": "params.json"})

    return {"status": "improved", "loss": loss.item(), "new_params": new_params}

@app.post("/evolve")
def evolve():
    res = train_and_update()
    supa.table("evolver_logs").insert({"timestamp": time.time(), "result": res}).execute()
    return res

# Run: uvicorn evolver:app --host 0.0.0.0 --port 8000
