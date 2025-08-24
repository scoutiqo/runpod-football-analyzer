# smooth.py
from collections import defaultdict
import numpy as np
from scipy.signal import savgol_filter

def _savgol(y, window=9, poly=2):
    if len(y) < window:
        window = max(3, len(y) // 2 * 2 + 1)  # nearest odd
    if len(y) < 3:
        return np.array(y, dtype=float)
    return savgol_filter(np.array(y, dtype=float), window_length=window, polyorder=poly, mode='interp')

def smooth_and_speed(tracks, have_metric=False):
    """
    Input: tracks list of dicts (player+ball mixed) with fields:
      - t, type, id?, x_m,y_m or x_px,y_px
    Output: new list where each series is Savitzky–Golay smoothed and has:
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
            # numerical derivative
            dt = np.gradient(t)
            vx = np.gradient(xs) / (dt + 1e-6)
            vy = np.gradient(ys) / (dt + 1e-6)
            sp = np.sqrt(vx*vx + vy*vy)  # m/s
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
            sp = np.sqrt(vx*vx + vy*vy)  # px/s
            for i, a in enumerate(arr):
                a2 = dict(a)
                a2['x_px'] = float(xs[i]); a2['y_px'] = float(ys[i])
                a2['speed_pxps'] = float(sp[i])
                out.append(a2)
    # keep input order roughly by time
    return sorted(out, key=lambda z: z['t'])
