# pitchctl.py
import numpy as np

def time_to_intercept(player, x, y, max_speed=7.0):  # m/s baseline
    # Simple model: straight-line at max_speed; refine with acceleration later
    dx = x - player["x_m"]; dy = y - player["y_m"]
    return np.hypot(dx, dy) / max_speed

def control_grid(players_home, players_away, grid=(52,34), pitch=(105,68)):
    """
    Return grid of P(home control) per cell using logistic on TTI difference.
    """
    W,H = pitch; gx,gy = grid
    xs = np.linspace(-W/2, W/2, gx); ys = np.linspace(-H/2, H/2, gy)
    G = np.zeros((gy,gx), dtype=np.float32)
    for iy,y in enumerate(ys):
        for ix,x in enumerate(xs):
            tH = min(time_to_intercept(p,x,y) for p in players_home) if players_home else 1e6
            tA = min(time_to_intercept(p,x,y) for p in players_away) if players_away else 1e6
            # probability home control
            delta = tA - tH
            p_home = 1.0/(1.0+np.exp(-1.5*delta))
            G[iy,ix]=p_home
    return G
