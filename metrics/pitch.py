# metrics/pitch.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

# FIFA std pitch (meters). We'll normalize to 105x68 even if broadcast differs.
PITCH_W, PITCH_H = 105.0, 68.0

@dataclass
class PitchCalib:
    # If H is None we are in normalized 0..1 space
    H: np.ndarray | None = None       # 3x3 homography, pixel->meter
    px_scale_m: float | None = None   # fallback meters-per-pixel (approx)

def px_to_pitch(xy_px: np.ndarray, calib: PitchCalib, img_wh: Tuple[int,int]) -> np.ndarray:
    """
    xy_px: (N,2) pixel coords. Returns (N,2) in meters within [0..105, 0..68] if possible,
           else normalized [0..1] multiplied by pitch dims.
    """
    W,H = img_wh
    if calib.H is not None:
        pts = np.concatenate([xy_px, np.ones((xy_px.shape[0],1))], axis=1)
        pitch = (pts @ calib.H.T); pitch = pitch[:,:2] / pitch[:,2:3]
        return np.clip(pitch, [0,0], [PITCH_W, PITCH_H])
    # fallback: normalize by frame and scale to pitch
    pitch = np.stack([xy_px[:,0]/W*PITCH_W, xy_px[:,1]/H*PITCH_H], axis=1)
    return np.clip(pitch, [0,0], [PITCH_W, PITCH_H])

# Zones & labels
def zone_label(x_m: float, y_m: float) -> str:
    # thirds & half-spaces (attacking left->right for "home", we'll handle flips later)
    third = "def" if x_m < 35 else ("mid" if x_m < 70 else "final")
    # vertical lanes: wide L, half-space L, central, half-space R, wide R
    lane_edges = [0, 13.6, 27.2, 40.8, 54.4, 68.0]  # 68/5 increments
    names = ["LW","LHS","CEN","RHS","RW"]
    lane = names[int(np.digitize([y_m], lane_edges)-1)]
    return f"{third}_{lane}"
