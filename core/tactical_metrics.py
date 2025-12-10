import json
import numpy as np
import cv2
import math
from pathlib import Path

INPUT_TRACKS = "runs/json/tracks.json"
OUTPUT_METRICS = "runs/json/advanced_metrics.json"
OUTPUT_DIR = Path("runs/viz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def measure_distance(p1, p2):
    # Calculate distance using Meters if available, else Pixels
    x1 = p1.get('x_m', p1.get('x', 0))
    y1 = p1.get('y_m', p1.get('y', 0))
    x2 = p2.get('x_m', p2.get('x', 0))
    y2 = p2.get('y_m', p2.get('y', 0))
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def main():
    print("🧠 Calculating VALUE METRICS (Pressure, Packing, Control)...")
    
    if not Path(INPUT_TRACKS).exists(): 
        print("❌ Tracks file not found.")
        return

    data = json.loads(Path(INPUT_TRACKS).read_text())
    frames = data.get('frames', [])
    
    if not frames:
        print("❌ No frames in tracks.")
        return
    
    # Metrics Storage
    tactical_summary = {
        "pressure_map": [], # Avg pressure per frame
        "packing_map": []   # Packing score per frame
    }
    
    for f in frames:
        ball = f.get('ball')
        players = f.get('players', [])
        
        frame_pressure_total = 0
        frame_packing_total = 0
        
        # 1. PRESSURE METRIC (For every player)
        # "pressing_action (pressure within X meters)"
        for p in players:
            team = p.get('team')
            if team == "unknown": continue
            
            opponents = [o for o in players if o.get('team') != team and o.get('team') != "unknown"]
            
            pressure_score = 0.0
            for opp in opponents:
                dist = measure_distance(p, opp)
                # Physics: Pressure drops off exponentially with distance
                # High pressure if < 3m
                if dist < 5.0: 
                    pressure_score += (1.0 / (dist + 0.5)) 
            
            # Normalize to 0-100 scale (roughly)
            pressure_val = min(100, pressure_score * 20)
            p['pressure_index'] = round(pressure_val, 1)
            
            if pressure_val > 50:
                frame_pressure_total += 1

        # 2. PACKING METRIC
        # "number of opponents bypassed"
        # We need the ball possessor
        if ball:
            # Find player closest to ball
            possessor = min(players, key=lambda p: measure_distance(p, ball), default=None)
            
            if possessor:
                dist_to_ball = measure_distance(possessor, ball)
                if dist_to_ball < 2.0: # Has possession
                    team = possessor.get('team')
                    opponents = [o for o in players if o.get('team') != team]
                    
                    # Direction heuristic: Assume attacking towards the side with fewer teammates?
                    # Or simple X-axis logic: "Opponents behind the ball"
                    # This is rudimentary without pitch control models, but effective.
                    
                    # Count opponents with X < Ball X (if attacking right) or X > Ball X
                    # We'll count opponents "behind" the ball relative to center
                    # Assuming possession is in opponent half
                    
                    packed_count = 0
                    bx = ball.get('x_m', ball.get('x'))
                    
                    for o in opponents:
                        ox = o.get('x_m', o.get('x'))
                        # Check if opponent is "out of play" (behind ball)
                        # We approximate this by simple X comparison for now
                        if (bx > 52.5 and ox < bx) or (bx < 52.5 and ox > bx):
                            packed_count += 1
                    
                    f['packing_value'] = packed_count
                    frame_packing_total = packed_count

        tactical_summary["pressure_map"].append(frame_pressure_total)
        tactical_summary["packing_map"].append(frame_packing_total)

    # Save Enriched Tracks (Overwrites input to add 'pressure_index' to players)
    Path(INPUT_TRACKS).write_text(json.dumps(data))
    
    # Save Summary Metrics for Charts
    Path(OUTPUT_METRICS).write_text(json.dumps(tactical_summary, indent=2))
    
    print(f"✅ Added Pressure & Packing to {len(frames)} frames.")
    print(f"   💾 Saved enriched tracks to {INPUT_TRACKS}")
    print(f"   💾 Saved charts data to {OUTPUT_METRICS}")

if __name__ == "__main__":
    main()
