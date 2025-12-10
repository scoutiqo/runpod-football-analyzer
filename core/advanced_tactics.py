import json
import numpy as np
from pathlib import Path
from scipy.spatial import Voronoi

# CONFIG
INPUT_TRACKS = "runs/json/tracks.json"
OUTPUT_METRICS = "runs/json/advanced_metrics.json"

# PITCH DIMENSIONS (Normalized 0-1 for calculation)
GOAL_LEFT = [0.0, 0.5]
GOAL_RIGHT = [1.0, 0.5]

def calculate_pitch_control(frame):
    """
    Calculates 'Space Ownership' for Team A vs Team B.
    Uses a simplified Voronoi / Influence model.
    """
    team_a_control = 0.0
    team_b_control = 0.0
    total_area = 0.0
    
    # Collect points
    points = []
    teams = []
    
    for p in frame['players']:
        if p['team'] == 'A':
            points.append([p['x'], p['y']])
            teams.append(0) # A
        elif p['team'] == 'B':
            points.append([p['x'], p['y']])
            teams.append(1) # B
            
    if len(points) < 4: return 0.5 # Neutral if no players
    
    # Generate Voronoi Regions (Simple Space Ownership)
    # In a full prod system, we'd use a dense grid + speed influence (Pitch Control)
    # For this MVP, we sample random points on the pitch and check closest player
    
    # Monte Carlo Sampling for Area Control
    samples = np.random.rand(100, 2) # 100 test points
    points = np.array(points)
    
    for s in samples:
        # Find closest player to this random spot
        dists = np.sum((points - s)**2, axis=1)
        closest_idx = np.argmin(dists)
        
        if teams[closest_idx] == 0:
            team_a_control += 1
        else:
            team_b_control += 1
            
    return round(team_a_control / 100.0, 2)

def calculate_packing(p_start, p_end, opponents):
    """
    Calculates how many opponents were 'Packed' (Bypassed) by a pass/run.
    A crucial metric for evaluating attack effectiveness.
    """
    bypassed = 0
    start_x = p_start['x']
    end_x = p_end['x']
    
    # Direction of play?
    # We assume Team A attacks Right (x=1) and Team B attacks Left (x=0)
    # This needs dynamic detection, but we'll use a heuristic based on ball movement
    direction = 1 if end_x > start_x else -1
    
    for opp in opponents:
        ox = opp['x']
        # If opponent was between start and end X, and is now 'behind' the ball
        if direction == 1:
            if start_x < ox < end_x: bypassed += 1
        else:
            if end_x < ox < start_x: bypassed += 1
            
    return bypassed

def analyze_phase(ball_x, team_possession):
    """
    Determines if we are in Build-Up, Progression, or Attack.
    """
    # Assuming Attack Direction (Simplification: A->Right, B->Left)
    if team_possession == 'A':
        if ball_x < 0.33: return "Build Up"
        if ball_x < 0.66: return "Progression"
        return "Final Third"
    else:
        if ball_x > 0.66: return "Build Up"
        if ball_x > 0.33: return "Progression"
        return "Final Third"

def main():
    print("🧠 Starting ADVANCED TACTICAL ANALYSIS...")
    
    if not Path(INPUT_TRACKS).exists():
        print("❌ No tracks found.")
        return

    data = json.loads(Path(INPUT_TRACKS).read_text())
    frames = data.get('frames', [])
    
    tactical_data = []
    
    # State
    current_possession = "None"
    
    for i, f in enumerate(frames):
        ball = f.get('ball')
        players = f.get('players', [])
        
        # 1. Pitch Control (Who owns the field?)
        control_a = calculate_pitch_control(f)
        
        # 2. Find Possession (Context)
        # Reuse logic or read from silver_labels if available
        # Simple proximity for now
        if ball:
            closest_p = min(players, key=lambda p: (p['x']-ball['x'])**2 + (p['y']-ball['y'])**2, default=None)
            if closest_p:
                dist = ((closest_p['x']-ball['x'])**2 + (closest_p['y']-ball['y'])**2)**0.5
                if dist < 0.05: current_possession = closest_p['team']
        
        # 3. Phase of Play
        phase = "Neutral"
        if ball:
            phase = analyze_phase(ball['x'], current_possession)

        # 4. Packing (If this is a pass event)
        # We need to link this to the Event Miner results for context
        # For now, we calculate 'Potential Packing' (Defenders between ball and goal)
        # Team A Goal is X=1, Team B Goal is X=0
        
        team_a_opponents = [p for p in players if p['team'] == 'B']
        team_b_opponents = [p for p in players if p['team'] == 'A']
        
        pack_score_a = 0
        pack_score_b = 0
        
        if ball:
            # How many B players are between Ball and Right Goal?
            pack_score_a = sum(1 for o in team_a_opponents if o['x'] > ball['x'])
            # How many A players are between Ball and Left Goal?
            pack_score_b = sum(1 for o in team_b_opponents if o['x'] < ball['x'])

        # Store Metrics
        frame_metrics = {
            "frame": i,
            "pitch_control_a": control_a,
            "phase": phase,
            "defenders_behind_ball_a": pack_score_a, # Low number = Good for defense, High = Good for attack
            "defenders_behind_ball_b": pack_score_b
        }
        tactical_data.append(frame_metrics)

    # Save
    Path(OUTPUT_METRICS).write_text(json.dumps(tactical_data))
    print(f"✅ Generated Advanced Tactics for {len(frames)} frames.")
    print(f"   💾 Saved to: {OUTPUT_METRICS}")

if __name__ == "__main__":
    main()
