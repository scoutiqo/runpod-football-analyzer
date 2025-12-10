import numpy as np
import math

# PRO CONFIG (The AI will tune these)
SHOT_VELOCITY_THRESH = 0.015
CROSS_ANGLE_THRESH = 0.5
PRESSURE_RADIUS = 0.05

def calculate_vector(p1, p2):
    return np.array([p2['x'] - p1['x'], p2['y'] - p1['y']])

def get_speed(p1, p2, fps=25):
    dist = math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
    return dist * fps

def detect_complex_event(segment, frames):
    # Analyze the physics of the ball DURING the gap between touches
    start_frame = segment['end']
    end_frame = segment['next_start']
    
    if end_frame - start_frame < 2: return None
    
    # Get ball trajectory
    b_start = frames[start_frame].get('ball', {})
    b_end = frames[end_frame].get('ball', {})
    
    if not b_start or not b_end: return None
    
    # 1. SHOT DETECTION
    # Vector towards goal? (Assuming normalized x=0 and x=1 are goals)
    # High velocity?
    vec = calculate_vector(b_start, b_end)
    speed = get_speed(b_start, b_end)
    
    # Moving towards left goal (x=0) or right goal (x=1)
    moving_to_goal = (b_start['x'] > 0.8 and b_end['x'] > b_start['x']) or                      (b_start['x'] < 0.2 and b_end['x'] < b_start['x'])
                     
    if moving_to_goal and speed > SHOT_VELOCITY_THRESH:
        return "shot"

    # 2. CROSS DETECTION
    # Moving from wing (y < 0.2 or y > 0.8) into box (central y, end x near goal)
    is_wing_start = (b_start['y'] < 0.2 or b_start['y'] > 0.8)
    is_box_end = (0.25 < b_end['y'] < 0.75) and (b_end['x'] < 0.15 or b_end['x'] > 0.85)
    
    if is_wing_start and is_box_end:
        return "cross"

    return None

def calculate_pressure(player, opponents):
    if not opponents: return 0.0
    # Sum of inverse distances
    pressure = 0.0
    px, py = player['x'], player['y']
    
    for opp in opponents:
        dist = math.hypot(px - opp['x'], py - opp['y'])
        if dist < PRESSURE_RADIUS:
            pressure += (1.0 / (dist + 0.01))
            
    return min(pressure, 10.0) # Cap it
