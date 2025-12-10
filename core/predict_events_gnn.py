import json
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
import sys
import os

# Import your model definition
sys.path.append(os.getcwd())
from core.gnn_model import TacticalGNN

# CONFIG
INPUT_TRACKS = "runs/json/tracks.json"
OUTPUT_EVENTS = "runs/json/predicted_events_learned.json"
MODEL_PATH = "models/gnn_tactical_brain.pt"

# Must match training classes exactly
CLASSES = ["PASS", "SHOT", "DUEL", "TACKLE", "INTERCEPTION", "OTHER"]

def get_node_features(player, ball):
    # [x, y, team_encoding]
    team_map = {'A': 1, 'B': -1, 'unknown': 0}
    team = team_map.get(player.get('team'), 0)
    x = player.get('x_m', 0)
    y = player.get('y_m', 0)
    return [x, y, float(team)]

def create_graph_from_frame(f):
    node_feats = []
    
    # Ball Node
    ball = f.get('ball')
    if ball and ball.get('x_m', -1) != -1:
        node_feats.append([ball['x_m'], ball['y_m'], 0])
    else:
        node_feats.append([0, 0, 0]) 

    # Player Nodes
    for p in f['players']:
        if p.get('x_m', -1) != -1:
            node_feats.append(get_node_features(p, ball))
            
    if not node_feats: return None
    
    x = torch.tensor(node_feats, dtype=torch.float)
    
    # Fully Connected Edge Index (Simplified for Inference Speed)
    # Or use Radius Graph if you installed torch_cluster
    # For now, we connect everyone to everyone (dense) for context
    num_nodes = x.shape[0]
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j: edge_index.append([i, j])
            
    if not edge_index: return None
    
    return Data(x=x, edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous())

def main():
    print("🧠 RUNNING GNN INFERENCE (PRO LEVEL)...")
    
    if not Path(INPUT_TRACKS).exists() or not Path(MODEL_PATH).exists():
        print("❌ Missing input files.")
        return

    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Assuming 3 features (x,y,team) and len(CLASSES) output
    model = TacticalGNN(num_node_features=3, num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    data = json.loads(Path(INPUT_TRACKS).read_text())
    frames = data['frames']
    
    predictions = []
    
    # Run Inference Frame by Frame
    # Optimization: Skip frames with no physics
    for i, f in enumerate(frames):
        if i % 2 != 0: continue # Analyze every 2nd frame to save time
        
        graph = create_graph_from_frame(f)
        if not graph: continue
        
        graph = graph.to(device)
        
        with torch.no_grad():
            out = model(graph)
            prob = torch.exp(out).max().item()
            pred_idx = out.argmax(dim=1).item()
            label = CLASSES[pred_idx]
            
            # Thresholding
            if prob > 0.4 and label != "OTHER":
                predictions.append({
                    "frame": f.get('frame', i),
                    "label": label,
                    "prob": prob
                })

    print(f"✅ GNN Detected {len(predictions)} Tactical Events.")
    Path(OUTPUT_EVENTS).write_text(json.dumps(predictions))

if __name__ == "__main__":
    main()
