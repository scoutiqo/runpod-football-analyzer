# shape.py
import numpy as np
from scipy.spatial import ConvexHull

def team_shape(points_xy):
    """
    points_xy: Nx2 (meters). Return hull vertices + width/height/area/compactness.
    """
    arr = np.array(points_xy, dtype=float)
    if len(arr) < 3:
        return {"hull": [], "width_m":0, "height_m":0, "area_m2":0, "compactness":0}
    hull = ConvexHull(arr)
    poly = arr[hull.vertices]
    xs, ys = poly[:,0], poly[:,1]
    width = xs.max()-xs.min()
    height = ys.max()-ys.min()
    area = hull.area   # 2D: perimeter;  use hull.volume for area (scipy quirk)
    area = hull.volume
    perim = hull.area
    compact = (4*np.pi*area)/(perim**2 + 1e-6)
    return {"hull": poly.tolist(), "width_m":float(width), "height_m":float(height),
            "area_m2": float(area), "compactness": float(compact)}
