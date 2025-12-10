import json
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from tqdm import tqdm
import math

# CONFIG
INPUT_TRACKS = "runs/json/tracks.json"
OUTPUT_GRAPHS = "runs/graphs/match_graph.pt"
EDGE_RADIUS = 20.0 # Connect players within 20 meters (Influence Zone)

def get_node_features(player, ball, prev_player=None):
    # NODE FEATURES (What defines a player?)
    # [x, y, vx, vy, speed, team_encoding, has_ball]
    
    # 1. Position & Team
    x, y = player.get('x_m', 0), player.get('y_m', 0)
    team_map = {'A': 1, 'B': -1, 'unknown': 0}
    team = team_map.get(player.get('team'), 0)
    
    # 2. Velocity (If we have history)
    vx, vy, speed = 0, 0, 0
    if prev_player:
        dt = 0.04 # 25 FPS
        vx = (x - prev_player.get('x_m', 0)) / dt
        vy = (y - prev_player.get('y_m', 0)) / dt
        speed = math.hypot(vx, vy)
        
    # 3. Ball Context
    has_ball = 0
    if ball:
        bx, by = ball.get('x_m', 0), ball.get('y_m', 0)
        dist = math.hypot(x-bx, y-by)
        if dist < 1.5: has_ball = 1
        
    return [x, y, vx, vy, speed, float(team), float(has_ball)]

def create_frame_graph(frame_data, prev_frame_data=None):
    node_feats = []
    
    # 1. Build Player Nodes
    # We need a consistent ID map for velocity calculation
    prev_map = {p['id']: p for p in prev_frame_data['players']} if prev_frame_data else {}
    
    ball = frame_data.get('ball')
    
    for p in frame_data['players']:
        prev = prev_map.get(p['id'])
        feat = get_node_features(p, ball, prev)
        node_feats.append(feat)
        
    if not node_feats: return None
    
    x = torch.tensor(node_feats, dtype=torch.float)
    
    # 2. Build Edges (Spatial Connections)
    edge_index = []
    edge_attr = []
    
    num_nodes = x.shape[0]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j: continue
            
            # Calculate Distance
            pos_i = x[i, :2]
            pos_j = x[j, :2]
            dist = torch.norm(pos_i - pos_j).item()
            
            # If within influence radius, connect them
            if dist < EDGE_RADIUS:
                edge_index.append([i, j])
                # Edge Feature: [Distance]
                edge_attr.append([dist])
                
    if not edge_index: return None
    
    return Data(x=x, 
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float))

def main():
    print("🕸️ CONVERTING TRACKS TO TACTICAL GRAPHS...")
    
    if not Path(INPUT_TRACKS).exists():
        print("❌ No tracks found.")
        return
        
    data = json.loads(Path(INPUT_TRACKS).read_text())
    frames = data.get('frames', [])
    
    graph_list = []
    Path("runs/graphs").mkdir(parents=True, exist_ok=True)
    
    prev_frame = None
    for f in tqdm(frames):
        # Only process if we have physics (meters)
        if not f['players'] or f['players'][0].get('x_m', -1) == -1:
            prev_frame = f
            continue
            
        graph = create_frame_graph(f, prev_frame)
        if graph:
            graph_list.append(graph)
        prev_frame = f
            
    print(f"✅ Generated {len(graph_list)} Tactical Graphs.")
    
    # Save as PyTorch Dataset
    torch.save(graph_list, OUTPUT_GRAPHS)
    print(f"💾 Saved to {OUTPUT_GRAPHS}")

if __name__ == "__main__":
    main()
