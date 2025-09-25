# pitchctl.py
import numpy as np

def tti_accel_model(px, py, tx, ty, vmax=7.0, amax=3.0, t_react=0.3):
    """
    Reaction + accel limited time to intercept a target point.
    Closed-form approx: distance covered with accelerate->cruise profile.
    """
    dx = tx - px; dy = ty - py
    d = np.hypot(dx, dy)
    # distance to reach vmax under amax:
    d_accel = 0.5 * vmax * vmax / max(amax, 1e-6)
    t_accel = vmax / max(amax, 1e-6)
    t = np.where(
        d <= d_accel,
        # triangular profile: solve d = 0.5*a*t^2  → t = sqrt(2d/a)
        np.sqrt(2.0 * d / max(amax, 1e-6)),
        # trapezoidal: accel to vmax then cruise
        t_accel + (d - d_accel) / max(vmax, 1e-6)
    )
    return t + t_react

def control_grid(players_home, players_away, grid=(52,34), pitch=(105,68),
                 vmax=7.0, amax=3.0, t_react=0.3, k_logit=1.5):
    """
    Vectorized P(home control) using logistic on ΔTTI = t_away_min - t_home_min.
    """
    W, H = pitch; gx, gy = grid
    xs = np.linspace(-W/2, W/2, gx).astype(np.float32)
    ys = np.linspace(-H/2, H/2, gy).astype(np.float32)
    X, Y = np.meshgrid(xs, ys)     # (gy,gx)

    def min_tti(players):
        if not players:
            return np.full_like(X, 1e6, dtype=np.float32)
        PX = np.array([p["x_m"] for p in players], dtype=np.float32)[:, None, None]
        PY = np.array([p["y_m"] for p in players], dtype=np.float32)[:, None, None]
        t = tti_accel_model(PX, PY, X[None,...], Y[None,...], vmax=vmax, amax=amax, t_react=t_react)
        return t.min(axis=0).astype(np.float32)

    tH = min_tti(players_home)
    tA = min_tti(players_away)
    delta = tA - tH
    # logistic; tune k_logit by validation (≈1.0–2.5)
    P = 1.0 / (1.0 + np.exp(-k_logit * delta))
    return P.astype(np.float32)
