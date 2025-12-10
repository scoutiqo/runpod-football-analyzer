import json
import numpy as np
from pathlib import Path

# CONFIG
TRACKS_FILE = "runs/json/tracks.json"
CHAINS_FILE = "runs/json/possession_chains.json"
OUTPUT_FILE = "runs/json/tactical_events.json"

# DEFINITIONS (Meters)
PITCH_WIDTH = 68.0
SWITCH_THRESHOLD = 35.0     # A pass must travel 35m laterally
HIGH_PRESS_HEIGHT = 70.0    # Pressing must happen in final 35m of pitch

def main():
    print("🧠 Detecting TACTICAL PATTERNS (Switches, Pressing, Line-Breaks)...")
    
    if not Path(TRACKS_FILE).exists() or not Path(CHAINS_FILE).exists():
        return

    tracks = json.loads(Path(TRACKS_FILE).read_text())['frames']
    chains = json.loads(Path(CHAINS_FILE).read_text())
    
    tactical_events = []

    for chain in chains:
        team = chain['team']
        events = chain['events']
        
        # Analyze every Pass in the chain
        for i, evt in enumerate(events):
            if 'pass' not in evt['label'].lower(): continue
            
            # Get start and end frames of the pass
            f_start = evt['frame']
            # Heuristic: The pass "ends" when the next event starts
            f_end = events[i+1]['frame'] if i+1 < len(events) else chain.get('end_frame', f_start+50)
            
            try:
                start_ball = tracks[f_start]['ball']
                end_ball = tracks[f_end]['ball']
                
                # 1. SWITCH OF PLAY (Horizontal Distance)
                # Calculate Delta Y (Width change)
                width_covered = abs(start_ball['y_m'] - end_ball['y_m'])
                if width_covered > SWITCH_THRESHOLD:
                    evt['tactical_tag'] = "switch_of_play"
                    tactical_events.append(evt)

                # 2. LINE-BREAKING PASS (Packing)
                # We check the 'packing_value' we calculated earlier
                packing = tracks[f_start].get('packing_value', 0)
                if packing >= 3: # Bypassed 3+ defenders
                    evt['tactical_tag'] = "line_breaking_pass"
                    tactical_events.append(evt)
                    
            except: pass

    # 3. HIGH PRESS DETECTION (Frame by Frame)
    # We look for frames where 3+ players of Team A are sprinting in Team B's defensive third
    for f_idx, frame in enumerate(tracks):
        if f_idx % 10 != 0: continue # Optimization
        
        for team in ['A', 'B']:
            # Define Attacking Third (Simple X check)
            # Assuming A attacks Right (X > 70), B attacks Left (X < 35)
            is_attacking_third = False
            pressers = 0
            
            for p in frame['players']:
                if p['team'] != team: continue
                
                # Check Position
                x = p.get('x_m', 0)
                if team == 'A' and x > HIGH_PRESS_HEIGHT: is_attacking_third = True
                elif team == 'B' and x < (105 - HIGH_PRESS_HEIGHT): is_attacking_third = True
                
                # Check Intensity (Speed > 15 km/h is a run/press)
                if is_attacking_third and p.get('speed', 0) > 15.0:
                    pressers += 1
            
            if pressers >= 3:
                tactical_events.append({
                    "frame": f_idx,
                    "label": "high_press",
                    "team": team,
                    "players_involved": pressers
                })

    # Save
    Path(OUTPUT_FILE).write_text(json.dumps(tactical_events, indent=2))
    print(f"✅ Detected {len(tactical_events)} Tactical Patterns.")

if __name__ == "__main__":
    main()
