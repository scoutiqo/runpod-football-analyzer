# shape.py
import numpy as np
from scipy.spatial import ConvexHull

def _oriented_extent(poly):
    # PCA for principal axes
    mu = poly.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(poly - mu, full_matrices=False)
    axes = Vt  # 2x2
    proj = (poly - mu) @ axes.T
    w = proj[:,0].max() - proj[:,0].min()
    h = proj[:,1].max() - proj[:,1].min()
    return float(w), float(h)

def team_shape(points_xy):
    arr = np.array(points_xy, dtype=float)
    if len(arr) < 3:
        return {"hull": [], "width_m":0, "height_m":0,
                "width_oriented_m":0, "height_oriented_m":0,
                "area_m2":0, "compactness":0}
    hull = ConvexHull(arr)
    poly = arr[hull.vertices]
    xs, ys = poly[:,0], poly[:,1]
    width = float(xs.max()-xs.min())
    height = float(ys.max()-ys.min())
    area = float(hull.volume)    # area in 2D
    perim = float(hull.area)     # perimeter in 2D
    compact = (4*np.pi*area)/((perim**2)+1e-6)
    w_o, h_o = _oriented_extent(poly)
    return {"hull": poly.tolist(), "width_m":width, "height_m":height,
            "width_oriented_m":w_o, "height_oriented_m":h_o,
            "area_m2": area, "compactness": compact}
