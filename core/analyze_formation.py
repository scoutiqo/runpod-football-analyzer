import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans

INPUT_TRACKS = "runs/json/tracks.json"
OUTPUT_FORM = "runs/json/formation.json"

def get_formation_string(positions):
    # positions = list of avg_x for the TOP 10 players
    if not positions: return "Unknown"
    
    try:
        # Cluster into 3 lines (Def, Mid, Att)
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        X = np.array(positions).reshape(-1, 1)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_.flatten()
        
        sorted_indices = np.argsort(centers)
        counts = [0, 0, 0]
        for l in labels:
            pos_idx = np.where(sorted_indices == l)[0][0]
            counts[pos_idx] += 1
            
        return f"{counts[0]}-{counts[1]}-{counts[2]}"
    except: return "Unknown"

def main():
    print("📍 Analyzing Formation (Forced Top 11)...")
    if not Path(INPUT_TRACKS).exists(): return
    data = json.loads(Path(INPUT_TRACKS).read_text())
    
    team_data = {'A': defaultdict(list), 'B': defaultdict(list)}
    
    for f in data.get('frames', []):
        for p in f.get('players', []):
            team = p.get('team')
            # Use meters if available, else normalized
            x = p.get('x_m', -1)
            if x == -1: x = p.get('x', 0) * 105
            
            if team in ['A', 'B'] and x > 0:
                team_data[team][p['id']].append(x)

    formations = {}
    
    for team in ['A', 'B']:
        players = team_data[team]
        roster = []
        
        for pid, x_vals in players.items():
            # Must be present for > 2 seconds
            if len(x_vals) > 50:
                roster.append({'id': pid, 'avg_x': np.mean(x_vals), 'count': len(x_vals)})
        
        # KEY FIX: Sort by duration and TAKE TOP 11
        roster.sort(key=lambda x: x['count'], reverse=True)
        starters = roster[:11] # Force 11 players
        
        # Sort by X to separate GK
        starters.sort(key=lambda x: x['avg_x'])
        
        if len(starters) > 1:
            # Assume first is GK, use rest for formation
            field_players = [p['avg_x'] for p in starters[1:]]
            formations[team] = get_formation_string(field_players)
        else:
            formations[team] = "Unknown"

    Path(OUTPUT_FORM).write_text(json.dumps(formations, indent=2))
    print(f"✅ Fixed Formations: {formations}")

if __name__ == "__main__":
    main()
