#!/usr/bin/env python
"""
Evaluate calibration quality for a tracks_meters_*.json file.

Metrics:
  - total points with meter coords
  - in-pitch ratio (fraction of points inside [0, L] x [0, W])
  - x_m, y_m ranges
  - speed distribution over player points (P50, P95, max) in m/s

Usage:
  python tools/evaluate_calibration.py \
    --in runs/json/tracks_meters_short.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple


def load_tracks(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    tracks = data.get("tracks", [])
    pitch = data.get("pitch", {})
    length = float(pitch.get("length_m", 105.0))
    width = float(pitch.get("width_m", 68.0))
    return tracks, length, width


def compute_in_pitch_stats(tracks, length: float, width: float) -> Tuple[int, int, float, Tuple[float, float], Tuple[float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    in_pitch = 0
    total = 0

    for r in tracks:
        if "x_m" not in r or "y_m" not in r:
            continue
        x = float(r["x_m"])
        y = float(r["y_m"])
        xs.append(x)
        ys.append(y)
        total += 1
        if 0.0 <= x <= length and 0.0 <= y <= width:
            in_pitch += 1

    if not xs:
        return 0, 0, 0.0, (0.0, 0.0), (0.0, 0.0)

    in_ratio = in_pitch / max(total, 1)
    return total, in_pitch, in_ratio, (min(xs), max(xs)), (min(ys), max(ys))


def compute_speed_stats(tracks) -> Tuple[int, float, float, float]:
    """
    Very simple speed estimate:
      - group by (type, id)
      - sort by t
      - compute dist / dt between consecutive points (dt in (0, 2] seconds)
    Returns: (n_speeds, p50, p95, vmax)
    """
    by_key = {}
    for r in tracks:
        if "x_m" not in r or "y_m" not in r:
            continue
        t = float(r.get("t", 0.0))
        typ = r.get("type", "unknown")
        pid = r.get("id", -1)
        key = (typ, pid)
        by_key.setdefault(key, []).append((t, float(r["x_m"]), float(r["y_m"])))

    speeds: List[float] = []
    for (typ, pid), pts in by_key.items():
        if typ != "player":
            continue
        if len(pts) < 2:
            continue
        pts.sort(key=lambda p: p[0])
        last_t, last_x, last_y = pts[0]
        for t, x, y in pts[1:]:
            dt = t - last_t
            if dt <= 0.0 or dt > 2.0:
                last_t, last_x, last_y = t, x, y
                continue
            dx = x - last_x
            dy = y - last_y
            dist = math.hypot(dx, dy)
            v = dist / dt
            speeds.append(v)
            last_t, last_x, last_y = t, x, y

    if not speeds:
        return 0, 0.0, 0.0, 0.0

    speeds_sorted = sorted(speeds)
    n = len(speeds_sorted)

    def percentile(p: float) -> float:
        if n == 0:
            return 0.0
        idx = min(int(p * (n - 1)), n - 1)
        return speeds_sorted[idx]

    p50 = percentile(0.5)
    p95 = percentile(0.95)
    vmax = speeds_sorted[-1]
    return n, p50, p95, vmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_path",
        default="runs/json/tracks_meters_short.json",
        help="Path to tracks_meters_*.json",
    )
    args = ap.parse_args()

    path = Path(args.in_path)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    tracks, length, width = load_tracks(path)

    print(f"[eval_calib] File: {path}")
    print(f"[eval_calib] Pitch: length={length:.2f}m, width={width:.2f}m")

    total, in_pitch, in_ratio, (xmin, xmax), (ymin, ymax) = compute_in_pitch_stats(tracks, length, width)
    print(f"[eval_calib] Points with meter coords: {total}")
    print(f"[eval_calib] In-pitch points: {in_pitch} ({in_ratio*100:.1f}%)")
    print(f"[eval_calib] x_m range: {xmin:.3f} → {xmax:.3f}")
    print(f"[eval_calib] y_m range: {ymin:.3f} → {ymax:.3f}")

    n_speeds, p50, p95, vmax = compute_speed_stats(tracks)
    print(f"[eval_calib] Speed samples (player only): {n_speeds}")
    if n_speeds > 0:
        print(f"[eval_calib] Speeds m/s: P50={p50:.3f}, P95={p95:.3f}, max={vmax:.3f}")
    else:
        print("[eval_calib] No speed samples (check ids / type=='player').")


if __name__ == "__main__":
    main()
