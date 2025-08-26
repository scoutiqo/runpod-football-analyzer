# smooth.py
from collections import defaultdict
import numpy as np
from scipy.signal import savgol_filter

def _savgol(y, window=9, poly=2):
    """
    Safe Savitzky–Golay:
    - never uses a window > series length
    - enforces odd window
    - clamps poly < window
    - falls back to raw series on any error
    """
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    if n < 3:
        return y.copy()

    # window ≤ n and odd
    w = min(int(window), n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return y.copy()

    p = int(min(poly, w - 1))
    if p < 1:
        p = 1
    if p >= w:  # still not valid -> bail out
        return y.copy()

    try:
        return savgol_filter(y, window_length=w, polyorder=p, mode="interp")
    except Exception:
        return y.copy()

def smooth_and_speed(tracks, have_metric=False):
    """
    Input: tracks list of dicts (player+ball mixed) with fields:
      - t, type, id?, x_m,y_m or x_px,y_px
    Output: list with smoothed positions + speed:
      - speed_ms (if metric) else speed_pxps
    """
    by_key = defaultdict(list)
    for r in tracks:
        key = ('ball', None) if r['type'] == 'ball' else ('player', r['id'])
        by_key[key].append(r)

    out = []
    for (kind, pid), arr in by_key.items():
        arr = sorted(arr, key=lambda z: z['t'])
        t  = np.array([a['t'] for a in arr], dtype=float)

        if have_metric and 'x_m' in arr[0] and 'y_m' in arr[0]:
            x = np.array([a.get('x_m', np.nan) for a in arr], dtype=float)
            y = np.array([a.get('y_m', np.nan) for a in arr], dtype=float)
            xs = _savgol(x); ys = _savgol(y)
            dt = np.gradient(t)
            vx = np.gradient(xs) / (dt + 1e-6)
            vy = np.gradient(ys) / (dt + 1e-6)
            sp = np.sqrt(vx * vx + vy * vy)  # m/s
            for i, a in enumerate(arr):
                a2 = dict(a)
                a2['x_m'] = float(xs[i]); a2['y_m'] = float(ys[i])
                a2['speed_ms'] = float(sp[i])
                out.append(a2)
        else:
            x = np.array([a.get('x_px', np.nan) for a in arr], dtype=float)
            y = np.array([a.get('y_px', np.nan) for a in arr], dtype=float)
            xs = _savgol(x); ys = _savgol(y)
            dt = np.gradient(t)
            vx = np.gradient(xs) / (dt + 1e-6)
            vy = np.gradient(ys) / (dt + 1e-6)
            sp = np.sqrt(vx * vx + vy * vy)  # px/s
            for i, a in enumerate(arr):
                a2 = dict(a)
                a2['x_px'] = float(xs[i]); a2['y_px'] = float(ys[i])
                a2['speed_pxps'] = float(sp[i])
                out.append(a2)

    return sorted(out, key=lambda z: z['t'])
