#!/usr/bin/env python
import json
import numpy as np
import argparse
from pathlib import Path
from sklearn.cluster import DBSCAN

# DEFAULTS
DEFAULT_INPUT = "runs/json/formatted_tracks_silver.json"
DEFAULT_OUTPUT = "runs/json/formatted_tracks_silver.json"

# CONFIG
MIN_TRACK_DURATION_FRAMES = 25
BOTTOM_HARD_CUTOFF = 0.92 
DBSCAN_EPS = 0.15
MIN_SAMPLES = 3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input tracks file", default=DEFAULT_INPUT)
    parser.add_argument("--output", help="Output cleaned file", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"🧹 Starting CLEANER on: {args.input}")
    
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return

    tracks_data = json.loads(Path(args.input).read_text())
    frames = tracks_data.get('frames', [])
    
    print(f"   Loaded {len(frames)} frames to clean.") # <--- THIS SHOULD MATCH YOUR VIDEO LENGTH
    
    # 1. LIFETIME ANALYSIS
    track_lifetimes = {}
    for f in frames:
        for p in f.get('players', []):
            pid = str(p['id'])
            track_lifetimes[pid] = track_lifetimes.get(pid, 0) + 1
            
    ghost_ids = {pid for pid, count in track_lifetimes.items() if count < MIN_TRACK_DURATION_FRAMES}

    total_kept = 0
    cleaned_frames = []
    
    for f in frames:
        players = f.get('players', [])
        valid_players = []
        
        # Normalize & Filter
        candidates = []
        for p in players:
            x_raw, y_raw = p.get('x', 0), p.get('y', 0)
            
            # Auto-Normalize
            if y_raw > 2.0: 
                y_norm = y_raw / 1080.0
                x_norm = x_raw / 1920.0
            else:
                y_norm, x_norm = y_raw, x_raw
                
            p['_x_norm'] = x_norm
            p['_y_norm'] = y_norm
            
            # Filter
            if str(p['id']) in ghost_ids: continue
            if y_norm > BOTTOM_HARD_CUTOFF: continue
            
            candidates.append(p)

        # Density Clustering
        if not candidates:
             f['players'] = []
             cleaned_frames.append(f)
             continue
             
        coords = np.array([[p['_x_norm'], p['_y_norm']] for p in candidates])
        
        if len(coords) > 5:
            clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=MIN_SAMPLES).fit(coords)
            labels = clustering.labels_
            
            counts = {}
            for lbl in labels:
                if lbl != -1: counts[lbl] = counts.get(lbl, 0) + 1
            
            if counts:
                main_cluster = max(counts, key=counts.get)
                for i, p in enumerate(candidates):
                    if labels[i] == main_cluster or (labels[i] == -1 and p['_y_norm'] < 0.80):
                        valid_players.append(p)
            else:
                valid_players = candidates
        else:
            valid_players = candidates
            
        # Cleanup temp keys
        for p in valid_players:
            if '_x_norm' in p: del p['_x_norm']
            if '_y_norm' in p: del p['_y_norm']
            
        total_kept += len(valid_players)
        cleaned_frames.append({"t": f['t'], "ball": f.get('ball'), "players": valid_players})

    tracks_data['frames'] = cleaned_frames
    Path(args.output).write_text(json.dumps(tracks_data))
    print(f"✅ Cleaned {len(frames)} frames. Saved to {args.output}")

if __name__ == '__main__':
    main()
