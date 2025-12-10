import json
import numpy as np
import math
from pathlib import Path
from collections import defaultdict

# CONFIG
INPUT_PREDS = "runs/json/predicted_events_learned.json"
INPUT_TRACKS = "runs/json/tracks.json"
OUTPUT_FILE = "runs/json/final_events_viewer.json"
CONFIDENCE_THRESH = 0.25 
FPS = 25.0
MERGE_WINDOW = 2.0 

EVENT_CATEGORIES = {
    "shot": "shot", "shot_foot": "shot", "shot_header": "shot", "volley": "shot", "penalty_shot": "shot", "goal": "goal",
    "pass": "pass", "short_pass": "pass", "long_ball": "pass", "cross": "cross", "cutback": "cross", "through_ball": "pass",
    "duel": "duel", "ground_duel": "duel", "aerial_duel": "duel", "tackle": "tackle", "sliding_tackle": "tackle",
    "interception": "interception", "block": "block", "clearance": "clearance", "save": "save",
    "corner": "corner", "corner_taken": "corner", "free_kick": "free_kick", "goal_kick": "goal_kick", "foul": "foul", "offside": "offside"
}

def get_event_team(frame_idx, tracks):
    if frame_idx >= len(tracks): return "Unknown"
    
    frame_data = tracks[frame_idx]
    ball = frame_data.get('ball')
    if not ball: return "Unknown"
    
    # Find closest player
    bx, by = ball.get('x', 0), ball.get('y', 0)
    min_dist = 1.0
    closest_team = "Unknown"
    
    for p in frame_data.get('players', []):
        px, py = p.get('x', 0), p.get('y', 0)
        dist = math.hypot(bx-px, by-py)
        if dist < min_dist:
            min_dist = dist
            closest_team = p.get('team', 'Unknown')
            
    return closest_team

def main():
    print("📦 Sequencing SMART Events (With Team Identity)...")
    
    if not Path(INPUT_PREDS).exists(): return
    preds = json.loads(Path(INPUT_PREDS).read_text())
    
    tracks = []
    if Path(INPUT_TRACKS).exists():
        try:
            tracks = json.loads(Path(INPUT_TRACKS).read_text())['frames']
        except: pass

    # 1. Pre-Filter
    candidates = []
    for p in preds:
        if p['prob'] < CONFIDENCE_THRESH: continue
        label = p['label']
        if label not in EVENT_CATEGORIES: continue
        candidates.append(p)
        
    candidates.sort(key=lambda x: x['prob'], reverse=True)
    
    final_events = []
    occupied_frames = set()
    window_frames = int(MERGE_WINDOW * FPS)
    
    # 2. NMS
    for c in candidates:
        f = c['frame']
        is_clashed = False
        for neighbor in range(f - window_frames, f + window_frames):
            if neighbor in occupied_frames:
                is_clashed = True
                break
        
        if not is_clashed:
            # ENRICH WITH TEAM
            c['team'] = get_event_team(f, tracks)
            final_events.append(c)
            for i in range(f - window_frames, f + window_frames):
                occupied_frames.add(i)
                
    final_events.sort(key=lambda x: x['frame'])
    
    # 3. Format
    output = []
    stats = defaultdict(int)
    
    for e in final_events:
        raw_label = e['label']
        ui_label = EVENT_CATEGORIES.get(raw_label, "other")
        t = e['frame'] / FPS
        
        output.append({
            "id": f"{ui_label}_{e['frame']}",
            "frame": e['frame'],
            "time": round(t, 2),
            "start": round(max(0, t - 4), 2),
            "end": round(t + 4, 2),
            "label": ui_label.upper(),
            "detail": raw_label,
            "conf": round(e['prob'], 2),
            "team": e.get('team', 'Unknown') # Now populated
        })
        stats[ui_label] += 1

    Path(OUTPUT_FILE).write_text(json.dumps(output, indent=2))
    print(f"✅ Exported {len(output)} clips with Team IDs.")
    print(f"   Breakdown: {dict(stats)}")

if __name__ == "__main__":
    main()
