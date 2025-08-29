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

    w = min(int(window), n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return y.copy()

    p = int(min(poly, w - 1))
    if p < 1:
        p = 1
    if p >= w:
        return y.copy()

    try:
        return savgol_filter(y, window_length=w, polyorder=p, mode="interp")
    except Exception:
        return y.copy()

def _valid_time_and_xy(t, x, y):
    """
    Drop NaNs and duplicated times; return filtered arrays and an index map.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # mask finite
    m = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)

    t2 = t[m]
    x2 = x[m]
    y2 = y[m]
    idx = np.nonzero(m)[0]

    if t2.size == 0:
        return t2, x2, y2, idx

    # ensure strictly increasing time: keep first of duplicates
    # (np.diff==0 means duplicates)
    keep = np.ones_like(t2, dtype=bool)
    if t2.size > 1:
        keep[1:] = np.diff(t2) != 0.0
    t3 = t2[keep]
    x3 = x2[keep]
    y3 = y2[keep]
    idx2 = idx[keep]

    return t3, x3, y3, idx2

def _safe_speed(t, xs, ys):
    """
    Compute speed with guards. Returns (speed, dt) aligned with xs/ys.
    """
    t = np.asarray(t, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    # Need at least 2 points to define a gradient.
    if t.size < 2 or xs.size < 2 or ys.size < 2:
        return np.zeros_like(xs), np.zeros_like(xs)

    dt = np.gradient(t)
    # Guard against zeros/negatives due to weird timestamps
    dt = np.where(np.abs(dt) < 1e-6, 1e-6, dt)

    vx = np.gradient(xs) / dt
    vy = np.gradient(ys) / dt
    sp = np.sqrt(vx * vx + vy * vy)
    return sp, dt

def smooth_and_speed(tracks, have_metric=False):
    """
    Input: tracks list of dicts (player+ball mixed) with fields:
      - t, type, id?, x_m,y_m or x_px,y_px
    Output: list with smoothed positions + speed:
      - speed_ms (if metric) else speed_pxps
    """
    by_key = defaultdict(list)
    for r in tracks:
        key = ('ball', None) if r.get('type') == 'ball' else ('player', r.get('id'))
        by_key[key].append(r)

    out = []
    for (kind, pid), arr in by_key.items():
        arr = sorted(arr, key=lambda z: z.get('t', 0.0))
        t_raw = [a.get('t', np.nan) for a in arr]

        if have_metric and ('x_m' in arr[0] and 'y_m' in arr[0]):
            x_raw = [a.get('x_m', np.nan) for a in arr]
            y_raw = [a.get('y_m', np.nan) for a in arr]
        else:
            x_raw = [a.get('x_px', np.nan) for a in arr]
            y_raw = [a.get('y_px', np.nan) for a in arr]

        # Filter invalid/duplicate timestamps
        t, x, y, kept_idx = _valid_time_and_xy(t_raw, x_raw, y_raw)

        # Not enough valid samples → pass-through for those kept points
        if t.size < 2 or x.size < 2 or y.size < 2:
            for j, i in enumerate(kept_idx):
                a = dict(arr[i])
                if have_metric and ('x_m' in a and 'y_m' in a):
                    a['x_m'] = float(x[j]); a['y_m'] = float(y[j])
                    a['speed_ms'] = 0.0
                else:
                    a['x_px'] = float(x[j]); a['y_px'] = float(y[j])
                    a['speed_pxps'] = 0.0
                a['note'] = 'skipped smoothing: too few points'
                out.append(a)
            # if nothing valid at all, just keep originals with speed 0
            if t.size == 0:
                for a in arr:
                    a2 = dict(a)
                    if have_metric and ('x_m' in a2 and 'y_m' in a2):
                        a2['speed_ms'] = 0.0
                    else:
                        a2['speed_pxps'] = 0.0
                    a2['note'] = 'no valid samples'
                    out.append(a2)
            continue

        # Smooth only the valid subset
        xs = _savgol(x)
        ys = _savgol(y)

        sp, _ = _safe_speed(t, xs, ys)

        # Write back into kept indices
        for j, i in enumerate(kept_idx):
            a = dict(arr[i])
            if have_metric and ('x_m' in a and 'y_m' in a):
                a['x_m'] = float(xs[j]); a['y_m'] = float(ys[j])
                a['speed_ms'] = float(sp[j])
            else:
                a['x_px'] = float(xs[j]); a['y_px'] = float(ys[j])
                a['speed_pxps'] = float(sp[j])
            out.append(a)

        # Optionally, you could propagate zeros for dropped (invalid) indices,
        # but usually it's better to leave them out.

    return sorted(out, key=lambda z: z.get('t', 0.0))
