import json
import numpy as np
from pathlib import Path
import math

INPUT_FILE = "runs/json/tracks.json"

# CONFIG
MAX_TIME_GAP = 5.0    # Increased gap because Jerseys allow longer re-id
MAX_SPEED_M_S = 12.0  
FPS = 25.0

def calculate_distance(p1, p2):
    if p1.get('x_m', -1) != -1 and p2.get('x_m', -1) != -1:
        return math.hypot(p1['x_m'] - p2['x_m'], p1['y_m'] - p2['y_m'])
    return 1000.0 

def main():
    print("🧵 STARTING INTELLIGENT TRACK STITCHING (JERSEY AWARE)...")
    
    if not Path(INPUT_FILE).exists(): return
    data = json.loads(Path(INPUT_FILE).read_text())
    frames = data['frames']
    
    tracks_meta = {}
    
    # 1. Analyze Fragments
    for i, f in enumerate(frames):
        f_idx = f.get('frame', i)
        for p in f['players']:
            pid = str(p['id'])
            if pid not in tracks_meta:
                tracks_meta[pid] = {
                    "start_frame": f_idx, "end_frame": f_idx,
                    "start_pos": p, "end_pos": p,
                    "team": p['team'], "jersey": p.get('jersey_number'),
                    "frames": []
                }
            
            meta = tracks_meta[pid]
            meta['end_frame'] = f_idx
            meta['end_pos'] = p
            meta['frames'].append(f_idx)
            # Update jersey if found later in track
            if not meta['jersey'] and p.get('jersey_number'):
                meta['jersey'] = p.get('jersey_number')

    # 2. Stitching
    sorted_ids = sorted(tracks_meta.keys(), key=lambda k: tracks_meta[k]['start_frame'])
    merges = {} 
    active_tracks = [] 

    for pid in sorted_ids:
        current = tracks_meta[pid]
        
        # Filter active tracks
        active_tracks = [
            tid for tid in active_tracks 
            if (current['start_frame'] - tracks_meta[tid]['end_frame']) / FPS <= MAX_TIME_GAP
        ]
        
        best_match = None
        min_score = 9999.0
        
        # LOGIC:
        # A. If Jerseys Match -> Strongest Link
        # B. If Physics Match -> Secondary Link
        
        for candidate_id in active_tracks:
            cand = tracks_meta[candidate_id]
            
            # Team Check
            if current['team'] != cand['team'] and current['team'] != 'unknown' and cand['team'] != 'unknown':
                continue

            dt = (current['start_frame'] - cand['end_frame']) / FPS
            if dt <= 0: continue 

            # JERSEY MATCH CHECK
            if current['jersey'] and cand['jersey'] and current['jersey'] == cand['jersey']:
                best_match = candidate_id
                break # Found him! (Assume unique numbers per team)

            # PHYSICS CHECK (Fallback)
            dist = calculate_distance(cand['end_pos'], current['start_pos'])
            speed_req = dist / dt
            
            if speed_req < MAX_SPEED_M_S:
                if dist < min_score:
                    min_score = dist
                    best_match = candidate_id
        
        if best_match:
            root_id = best_match
            while root_id in merges: root_id = merges[root_id]
            merges[pid] = root_id
            
            # Propagate Jersey Number
            if current['jersey'] and not tracks_meta[root_id]['jersey']:
                tracks_meta[root_id]['jersey'] = current['jersey']
                
            tracks_meta[root_id]['end_frame'] = current['end_frame']
            tracks_meta[root_id]['end_pos'] = current['end_pos']
        else:
            active_tracks.append(pid)

    print(f"   ✅ Merged {len(merges)} broken tracks.")
    
    # 3. Rewrite Data
    unique_players = set()
    for f in frames:
        for p in f['players']:
            final_id = str(p['id'])
            while final_id in merges: final_id = merges[final_id]
            
            p['id'] = final_id
            
            # Ensure Jersey # is propagated to all frames
            if final_id in tracks_meta and tracks_meta[final_id]['jersey']:
                p['jersey_number'] = tracks_meta[final_id]['jersey']
                
            unique_players.add(final_id)
            
    print(f"   📉 Reduced {len(sorted_ids)} to {len(unique_players)} unique players.")
    Path(INPUT_FILE).write_text(json.dumps(data))

if __name__ == "__main__":
    main()
