# metrics/value_models.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import math
from metrics.xt import xt_delta, xt_value

def epv_lite_prob(ball_xy, pressure_n: int, nearest_def_m: float):
    """
    Very light EPV proxy: probability of finishing a possession in a shot within N seconds,
    based on xT at location and defensive pressure.
    """
    base = xt_value(*ball_xy) * 4.0     # scale to ~[0..0.48]
    press_pen = np.exp(-0.45*pressure_n) * (1.0 / (1.0 + np.exp(-(nearest_def_m-2.0))))
    p = np.clip(base * press_pen, 0.0, 0.6)
    return float(p)

def vaep_like(action_before_prob_sc, action_after_prob_sc, action_before_prob_conc, action_after_prob_conc):
    """
    VAEP idea: value(action) = (P(score after) - P(score before)) - (P(concede after) - P(concede before))
    Here we expose simple hooks; you can drive them from epv_lite or team xT potential.
    """
    return (action_after_prob_sc - action_before_prob_sc) - (action_after_prob_conc - action_before_prob_conc)

def value_pass(start_xy, end_xy, pressure_n, nearest_def_m):
    dxT = xt_delta(start_xy, end_xy)
    epv = epv_lite_prob(end_xy, pressure_n, nearest_def_m)
    return {"xT": dxT, "EPV": epv, "VAEP": dxT - 0.25*(pressure_n>1)}

def compute_values(events, config):
    """
    Compute value models for events:
    - xT (Expected Threat)
    - EPV (Expected Possession Value) 
    - VAEP (Value Added by Expected Possession)
    - Packing Score
    """
    enriched_events = []
    
    for event in events:
        enriched = event.copy()
        
        # xT calculation (simplified)
        enriched['xT'] = calculate_xt(event)
        
        # EPV calculation (simplified)
        enriched['EPV'] = calculate_epv(event)
        
        # VAEP calculation (simplified)
        enriched['VAEP'] = calculate_vaep(event)
        
        # Packing score
        enriched['packing_score'] = calculate_packing_score(event)
        
        enriched_events.append(enriched)
    
    return enriched_events

def calculate_xt(event):
    """Calculate Expected Threat (xT) for an event"""
    # Simplified xT based on field position
    x, y = event.get('position', [0, 0])
    
    # Distance to goal
    distance_to_goal = math.sqrt((105 - x)**2 + (34 - y)**2)
    
    # xT decreases with distance from goal
    xt = max(0, 1.0 - (distance_to_goal / 100))
    
    return round(xt, 3)

def calculate_epv(event):
    """Calculate Expected Possession Value (EPV) for an event"""
    # Simplified EPV based on field position and event type
    x, y = event.get('position', [0, 0])
    
    # Base EPV from field position
    distance_to_goal = math.sqrt((105 - x)**2 + (34 - y)**2)
    base_epv = max(0, 1.0 - (distance_to_goal / 100))
    
    # Event type multiplier
    event_type = event.get('type', '')
    if event_type == 'shot':
        base_epv *= 2.0
    elif event_type == 'pass':
        base_epv *= 1.2
    
    return round(base_epv, 3)

def calculate_vaep(event):
    """Calculate Value Added by Expected Possession (VAEP) for an event"""
    # Simplified VAEP calculation
    xt = calculate_xt(event)
    epv = calculate_epv(event)
    
    # VAEP combines xT and EPV
    vaep = (xt + epv) / 2
    
    return round(vaep, 3)

def calculate_packing_score(event):
    """Calculate Packing Score for an event"""
    # Simplified packing score based on event type and position
    event_type = event.get('type', '')
    
    if event_type == 'pass':
        # Packing score for passes
        return 1
    elif event_type == 'shot':
        # Higher packing score for shots
        return 3
    else:
        return 0
