import json
import numpy as np
from pathlib import Path

INPUT_EVENTS = "runs/json/final_events_viewer.json"
INPUT_TRACKS = "runs/json/tracks.json"

# STANDARD xT GRID (12x8)
XT_GRID = np.array([
    [0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.014, 0.012],
    [0.008, 0.012, 0.016, 0.020, 0.025, 0.020, 0.016, 0.012],
    [0.010, 0.015, 0.022, 0.030, 0.035, 0.030, 0.022, 0.015],
    [0.014, 0.020, 0.035, 0.050, 0.060, 0.050, 0.035, 0.020],
    [0.018, 0.025, 0.045, 0.070, 0.090, 0.070, 0.045, 0.025],
    [0.022, 0.035, 0.060, 0.100, 0.150, 0.100, 0.060, 0.035],
    [0.025, 0.045, 0.080, 0.150, 0.300, 0.150, 0.080, 0.045],
    [0.022, 0.035, 0.060, 0.100, 0.150, 0.100, 0.060, 0.035],
    [0.018, 0.025, 0.045, 0.070, 0.090, 0.070, 0.045, 0.025],
    [0.014, 0.020, 0.035, 0.050, 0.060, 0.050, 0.035, 0.020],
    [0.010, 0.015, 0.022, 0.030, 0.035, 0.030, 0.022, 0.015],
    [0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.014, 0.012]
]).T

def get_xt_value(x, y):
    if x < 0 or y < 0: return 0.0
    col = int(min(x / (105/12), 11))
    row = int(min(y / (68/8), 7))
    try: return float(XT_GRID[row][col])
    except: return 0.0

def calculate_xg(x, y):
    # Distance to center of goal (105, 34)
    dist = np.sqrt((105 - x)**2 + (34 - y)**2)
    # Simple exponential decay model
    xg = 0.85 * np.exp(-0.15 * dist)
    return round(float(xg), 2)

def main():
    print("📈 Calculating xG and xThreat (With Fallback)...")
    
    if not Path(INPUT_EVENTS).exists() or not Path(INPUT_TRACKS).exists(): return
    
    events = json.loads(Path(INPUT_EVENTS).read_text())
    tracks = json.loads(Path(INPUT_TRACKS).read_text())['frames']
    
    count = 0
    for evt in events:
        f_idx = evt['frame']
        if f_idx >= len(tracks): continue
        
        ball = tracks[f_idx].get('ball')
        if not ball: continue
        
        bx = ball.get('x_m', -1)
        by = ball.get('y_m', -1)
        
        # --- FALLBACK FIX ---
        # If physics failed (-1), estimate from pixel coordinates
        if bx == -1:
            # Assume x=0 is 0m and x=1 is 105m
            bx = ball.get('x', 0) * 105.0
            by = ball.get('y', 0) * 68.0
        # --------------------
        
        if 'shot' in evt['label'].lower() or 'goal' in evt['label'].lower():
            evt['xg'] = calculate_xg(bx, by)
            
        evt['xt'] = get_xt_value(bx, by)
        count += 1
        
    Path(INPUT_EVENTS).write_text(json.dumps(events, indent=2))
    print(f"✅ Added Value Metrics to {count} events.")

if __name__ == "__main__":
    main()

