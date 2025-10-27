# metrics/xt.py
from __future__ import annotations
import numpy as np
from typing import Tuple

# Simple 12x8 xT surface (can be replaced by learned grid later).
# Here we generate a smooth heuristic increasing towards the center of the box & goal.
def xt_surface(nx=12, ny=8):
    xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny)
    X,Y = np.meshgrid(xs, ys, indexing="xy")
    # radial bias towards goal center (x=1, y=0.5), plus central lane bonus
    r = np.sqrt((1.0 - X)**2 + (Y-0.5)**2)
    surf = np.exp(-6*r) + 0.15*np.exp(-80*(Y-0.5)**2)
    # normalize to [0..1], then scale typical maxima (~0.12)
    surf = (surf - surf.min()) / (surf.max() - surf.min() + 1e-9)
    return surf * 0.12

def xt_value(x_m: float, y_m: float, pitch_w=105.0, pitch_h=68.0, S=None):
    if S is None: S = xt_surface()
    nx, ny = S.shape
    x = np.clip(x_m / pitch_w, 0, 1 - 1e-6); y = np.clip(y_m / pitch_h, 0, 1 - 1e-6)
    ix = min(nx-1, int(x*nx)); iy = min(ny-1, int(y*ny))
    return float(S[iy, ix])

def xt_delta(start_xy, end_xy, S=None):
    return xt_value(*end_xy, S=S) - xt_value(*start_xy, S=S)
