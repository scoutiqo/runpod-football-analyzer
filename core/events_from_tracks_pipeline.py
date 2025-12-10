#!/usr/bin/env python
import json
import math
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

# CONFIG
DEFAULT_INPUT = "runs/json/tracks.json"
LABELS_OUT = "runs/json/silver_labels.json"

# HYBRID THRESHOLDS (The Safety Net)
POSSESSION_DIST_METERS = 2.5    # Physics Check
POSSESSION_DIST_NORM = 0.06     # Visual Check (6% of screen width)
MIN_PASS_DIST = 3.0             # Meters
MIN_SHOT_SPEED = 15.0           # km/h

def get_dist(p1, p2, use_meters=False):
    """
    Calculates distance. 
    If use_meters is True, uses 'x_m'/'y_m'.
    Else uses 'x'/'y'.
    """
    if use_meters:
        # Check validity of meter coords
        if p1.get('x_m', -1) != -1 and p2.get('x_m', -1) != -1:
            return math.hypot(p1['x_m'] - p2['x_m'], p1['y_m'] - p2['y_m'])
            
    # Fallback to Normalized (Pixel-based)
    x1 = p1.get("x", 0); y1 = p1.get("y", 0)
    x2 = p2.get("x", 0); y2 = p2.get("y", 0)
    return math.hypot(x1 - x2, y1 - y2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input tracks JSON", default=DEFAULT_INPUT)
    args = parser.parse_args()

    print(f"⛏️ Mining Events (Hybrid Mode) from: {args.input}")
    
    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        return

    try:
        data = json.loads(Path(args.input).read_text())
        frames = data.get("frames", [])
    except:
        print("❌ Failed to read JSON.")
        return
    
    if not frames:
        print("❌ JSON has no frames.")
        return

    # Detect if we have valid physics data
    has_physics = False
    if frames[0].get('ball') and frames[0]['ball'].get('x_m', -1) != -1:
        has_physics = True

    segments = []
    curr_seg = None
    
    for i, fr in enumerate(frames):
        ball = fr.get("ball")
        players = fr.get("players") or []
        
        if not ball:
            if curr_seg: segments.append(curr_seg); curr_seg = None
            continue
            
        # HYBRID CHECK: Find closest player using BOTH metrics
        closest_p = None
        min_score = 9999.0
        
        for p in players:
            # 1. Visual Distance (Reliable)
            d_norm = get_dist(p, ball, use_meters=False)
            
            # 2. Physical Distance (Accurate ONLY if calibrated)
            d_phys = get_dist(p, ball, use_meters=True) if has_physics else 999
            
            # LOGIC: If either is close enough, it counts
            is_close = (d_norm < POSSESSION_DIST_NORM) or (has_physics and d_phys < POSSESSION_DIST_METERS)
            
            if is_close:
                # We prioritize visual closeness for selection
                if d_norm < min_score:
                    min_score = d_norm
                    closest_p = p
        
        if closest_p:
            pid = closest_p.get("id")
            team = closest_p.get("team", "unknown")
            
            # Save Ball Position
            bx = ball.get('x_m', 0) if has_physics else ball.get('x', 0)
            by = ball.get('y_m', 0) if has_physics else ball.get('y', 0)
            
            if curr_seg and curr_seg["pid"] == pid:
                curr_seg["end"] = i
                curr_seg["ball_end"] = {'x': bx, 'y': by}
            else:
                if curr_seg: segments.append(curr_seg)
                curr_seg = {
                    "pid": pid, 
                    "team": team, 
                    "start": i, 
                    "end": i, 
                    "ball_start": {'x': bx, 'y': by},
                    "ball_end": {'x': bx, 'y': by}
                }
        else:
             if curr_seg: segments.append(curr_seg); curr_seg = None

    if curr_seg: segments.append(curr_seg)
    print(f"   Found {len(segments)} possession segments.")
    
    # CLASSIFY EVENTS
    events = []
    stats = defaultdict(int)
    
    for k in range(len(segments) - 1):
        s1 = segments[k]
        s2 = segments[k+1]
        
        # Gap analysis
        gap = s2["start"] - s1["end"]
        if gap > 150: continue # Too long (5 seconds+)
        
        frame_idx = s1["end"] + min(5, gap // 2)
        
        # Calculate Distance Traveled
        dx = s2['ball_start']['x'] - s1['ball_end']['x']
        dy = s2['ball_start']['y'] - s1['ball_end']['y']
        dist = math.hypot(dx, dy)
        
        t1 = s1["team"]
        t2 = s2["team"]
        
        lbl = None
        
        if s1["pid"] != s2["pid"]:
            if t1 == t2 and t1 != "unknown":
                # PASS LOGIC
                # FIX: Variable name matched to definition at top
                req_dist = MIN_PASS_DIST if has_physics else 0.05
                
                if dist > req_dist:
                    lbl = "pass"
                else:
                    lbl = "short_pass" # New detailed label
            else:
                # DUEL LOGIC
                lbl = "duel" if gap < 25 else "ball_loss"
        else:
            # Same player (Carry)
            if dist > (5.0 if has_physics else 0.1):
                lbl = "ball_carry" # New detailed label
        
        if lbl:
            events.append({"frame": frame_idx, "label": lbl})
            stats[lbl] += 1
            
    print(f"✅ Generated {len(events)} Labels: {dict(stats)}")
    with open(LABELS_OUT, "w") as f:
        json.dump(events, f, indent=2)

if __name__ == "__main__":
    main()
