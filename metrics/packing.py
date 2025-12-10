# metrics/packing.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List

def count_packing(pass_start_xy, pass_end_xy, opp_xy: Dict[int,Tuple[float,float]], goal_x=105.0) -> int:
    """
    Bypassed defenders = opponents who were closer to their own goal than the ball at pass start,
    and are further from their goal than the receiving point (i.e., left behind the ball).
    Uses x-axis towards attacking goal (105m). Adjust sign if team attacks opposite.
    """
    x0, y0 = pass_start_xy; x1, y1 = pass_end_xy
    b = 0
    for _, (ox, oy) in opp_xy.items():
        # Opp considered "behind ball" if closer to own goal line than ball
        # For simplicity assume both teams attack to x=105; flip externally for away team.
        cond_before = ox > x0   # defender was nearer their own goal (to 105) than ball start
        cond_after  = ox <= x1  # now ball end is past the defender
        if cond_before and cond_after:
            b += 1
    return b
