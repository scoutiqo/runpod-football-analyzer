import json
import numpy as np
import math
from pathlib import Path
from collections import Counter

INPUT_TRACKS = "runs/json/tracks.json"
INPUT_EVENTS = "runs/json/final_events_viewer.json"
OUTPUT_CHAINS = "runs/json/possession_chains.json"

def get_closest_player(frame_data, ball_pos):
    if not ball_pos: return None
    
    use_meters = frame_data['players'][0].get('x_m', -1) != -1 if frame_data['players'] else False
    
    min_dist = 9999
    closest = None
    
    bx = ball_pos.get('x_m' if use_meters else 'x', 0)
    by = ball_pos.get('y_m' if use_meters else 'y', 0)

    for p in frame_data['players']:
        px = p.get('x_m' if use_meters else 'x', 0)
        py = p.get('y_m' if use_meters else 'y', 0)
        
        dist = math.sqrt((bx-px)**2 + (by-py)**2)
        if dist < min_dist:
            min_dist = dist
            closest = p
            
    # Robust Threshold: 8.0m for calibrated data, 0.15 for normalized
    thresh = 8.0 if use_meters else 0.15
    
    if min_dist < thresh:
        return closest
    return None

def classify_phase(start_x):
    # Standard Pitch is ~105m long.
    # Zone 1 (Defensive Third): 0 - 35m
    # Zone 2 (Middle Third): 35 - 70m
    # Zone 3 (Final Third): 70 - 105m
    
    # Note: This assumes strict Left->Right or Right->Left orientation.
    # Since we don't know which team is attacking which side yet, 
    # we categorize based on absolute field position for the Beta.
    
    if start_x < 35:
        return "Build-up"
    elif start_x < 70:
        return "Progression"
    else:
        return "Creation"

def main():
    print("🔗 BUILDING POSSESSION CHAINS (Tactical Phase Edition)...")
    
    if not Path(INPUT_TRACKS).exists() or not Path(INPUT_EVENTS).exists():
        print("❌ Input files missing.")
        return

    tracks_data = json.loads(Path(INPUT_TRACKS).read_text())
    tracks = tracks_data.get('frames', [])
    events = json.loads(Path(INPUT_EVENTS).read_text())
    
    chains = []
    events.sort(key=lambda x: x['frame'])
    
    current_chain = None
    
    for evt in events:
        f_idx = evt['frame']
        if f_idx >= len(tracks): continue
        
        # Robust Actor Lookup (+/- 5 frames)
        actor = None
        for offset in range(-5, 6):
            check_frame = f_idx + offset
            if 0 <= check_frame < len(tracks):
                ball = tracks[check_frame].get('ball')
                if not ball: continue
                candidate = get_closest_player(tracks[check_frame], ball)
                if candidate:
                    actor = candidate
                    break
        
        if actor:
            team = actor.get('team', 'unknown')
            pid = actor.get('id', 'unknown')
            
            evt['actor_id'] = pid
            evt['actor_team'] = team
            
            # Check Chain Continuity
            if current_chain is None or current_chain['team'] != team:
                # Close previous chain
                if current_chain:
                    current_chain['end_frame'] = f_idx
                    current_chain['players_involved'] = list(current_chain['players_involved'])
                    chains.append(current_chain)
                
                # Start new chain
                current_chain = {
                    "team": team,
                    "start_frame": f_idx,
                    "events": [evt],
                    "players_involved": {pid}
                }
            else:
                # Extend current chain
                current_chain['events'].append(evt)
                current_chain['players_involved'].add(pid)
                current_chain['end_frame'] = f_idx

    if current_chain:
        current_chain['players_involved'] = list(current_chain['players_involved'])
        chains.append(current_chain)

    # ENRICH WITH TACTICS
    for chain in chains:
        chain['duration_sec'] = round((chain.get('end_frame', 0) - chain['start_frame']) / 25.0, 2)
        chain['pass_count'] = sum(1 for e in chain['events'] if 'pass' in e.get('label', '').lower())
        
        # Get Physical Start Location
        try:
            start_ball = tracks[chain['start_frame']].get('ball', {})
            xm = start_ball.get('x_m', -1)
            
            if xm != -1:
                chain['start_x'] = round(xm, 1)
                chain['phase'] = classify_phase(xm)
            else:
                chain['phase'] = "Unknown" # Physics failed for this frame
        except:
            chain['phase'] = "Unknown"

    Path(OUTPUT_CHAINS).write_text(json.dumps(chains, indent=2))
    print(f"✅ Constructed {len(chains)} Possession Chains.")
    
    # Audit the Phases
    phases = [c.get('phase') for c in chains]
    print(f"   📊 Tactical Phases: {dict(Counter(phases))}")

if __name__ == "__main__":
    main()
