#!/usr/bin/env python
"""
event_features_v2.py

PRO-LEVEL FEATURE EXTRACTOR (v2.5)
----------------------------------
Generates 22 base physics/tactical features per frame, stacked into
~88-dim vectors for the Event Foundation Model.

Base Features (22):
  00. ball_x
  01. ball_y
  02. ball_vx (velocity x)
  03. ball_vy (velocity y)
  04. ball_speed (scalar)
  05. ball_accel (scalar)
  06. dist_team0_centroid
  07. dist_team1_centroid
  08. dist_nearest_any
  09. is_nearest_team0 (binary)
  10. dist_nearest_team0
  11. dist_nearest_team1
  12. pressure_index (sum of 1/dist for close players)
  13. close_players_cnt (within ~3m)
  14. team0_spread_x (compactness)
  15. team1_spread_x
  16. team0_spread_y (width)
  17. team1_spread_y
  18. ball_centrality_x (dist from center)
  19. ball_centrality_y (dist from center)
  20. packing_team0 (players behind ball)
  21. packing_team1
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

# --- CONFIGURATION ---
FIELD_WIDTH_M = 105.0
FIELD_HEIGHT_M = 68.0
# Normalized thresholds (assuming 1.0 = full field length/width)
DIST_CLOSE_THRESH = 3.0 / FIELD_WIDTH_M  # Approx 3 meters normalized
SMOOTHING_ALPHA = 0.3  # Simple exponential smoothing for velocity

def _load_tracks_json(path: str) -> Tuple[List[Dict[str, Any]], float]:
    """Load tracks JSON with robust fallback for fps and frame structure."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tracks file not found: {path}")
        
    data = json.loads(p.read_text(encoding="utf-8"))
    fps = 25.0
    frames: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        fps = float(data.get("fps", 25.0))
        if "frames" in data and isinstance(data["frames"], list):
            frames = data["frames"]
        elif "data" in data and isinstance(data["data"], list):
            frames = data["data"]
        else:
            # Fallback search
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    frames = v
                    break
    elif isinstance(data, list):
        frames = data

    if not frames:
        raise ValueError(f"No frames found in {path}")

    return frames, fps

def _extract_frame_objects(frame: Dict[str, Any]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Parses a frame to get the Ball dict and a List of Player dicts.
    Returns:
       ball: {'x': float, 'y': float, 'conf': float} (or None-like if missing)
       players: [{'x':, 'y':, 'team':}, ...]
    """
    # 1. Find Ball
    ball = {"x": -1.0, "y": -1.0, "found": False}
    
    # Try explicit ball key
    if isinstance(frame.get("ball"), dict):
        b = frame["ball"]
        ball["x"] = float(b.get("x", 0.5))
        ball["y"] = float(b.get("y", 0.5))
        ball["found"] = True
    
    # 2. Collect Players
    raw_objs = frame.get("players") or frame.get("objects") or frame.get("detections") or []
    players = []
    
    ball_candidates = []

    for obj in raw_objs:
        try:
            x = float(obj.get("x", obj.get("cx", 0.5)))
            y = float(obj.get("y", obj.get("cy", 0.5)))
            team = str(obj.get("team", "unknown"))
            label = str(obj.get("label", "")).lower()
            cls_name = str(obj.get("cls", "")).lower()
            is_ball_flag = obj.get("is_ball", False)
        except (ValueError, TypeError):
            continue

        # Check if this object is actually the ball (if explicit ball key failed or supplementary)
        is_ball = is_ball_flag or (label == "ball") or (cls_name == "ball")

        if is_ball:
            ball_candidates.append((x, y))
        else:
            players.append({"x": x, "y": y, "team": team})

    # Fallback: if 'ball' dict was missing but we found ball candidates in list
    if not ball["found"] and ball_candidates:
        ball["x"], ball["y"] = ball_candidates[0]
        ball["found"] = True
        
    return ball, players

def build_per_frame_base_features(tracks_path: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Main extraction loop. 
    Returns:
        feats: (n_frames, 22) matrix
        fps: float
        meta: dict
    """
    frames, fps = _load_tracks_json(tracks_path)
    n_frames = len(frames)
    
    # Identify Teams (Top 2 most frequent)
    team_counts = {}
    for fr in frames[::10]: # Sample every 10th frame
        _, players = _extract_frame_objects(fr)
        for p in players:
            t = p["team"]
            if t not in ["unknown", "referee", "other"]:
                team_counts[t] = team_counts.get(t, 0) + 1
    
    sorted_teams = sorted(team_counts.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_teams) >= 2:
        team0, team1 = sorted_teams[0][0], sorted_teams[1][0]
    else:
        team0, team1 = "0", "1" # Fallback

    # Feature Matrix
    # Cols: 22
    feats = np.zeros((n_frames, 22), dtype=np.float32)

    # State for kinematics
    prev_x, prev_y = 0.5, 0.5
    prev_vx, prev_vy = 0.0, 0.0
    
    # Pre-fill first valid position if possible
    # (Simple logic: linear scan to find first ball to init state)
    for fr in frames:
        b, _ = _extract_frame_objects(fr)
        if b["found"]:
            prev_x, prev_y = b["x"], b["y"]
            break

    for i, fr in enumerate(frames):
        ball, players = _extract_frame_objects(fr)
        
        # --- 1. Ball Physics (Smoothing & Handling Missing) ---
        if ball["found"]:
            bx, by = ball["x"], ball["y"]
        else:
            bx, by = prev_x + prev_vx, prev_y + prev_vy # Simple projection
            # Clamp
            bx = max(0.0, min(1.0, bx))
            by = max(0.0, min(1.0, by))

        # Calculate raw velocity
        raw_vx = bx - prev_x
        raw_vy = by - prev_y
        
        # Smooth velocity
        vx = prev_vx * (1 - SMOOTHING_ALPHA) + raw_vx * SMOOTHING_ALPHA
        vy = prev_vy * (1 - SMOOTHING_ALPHA) + raw_vy * SMOOTHING_ALPHA
        
        speed = (vx**2 + vy**2)**0.5
        accel = speed - (feats[i-1, 4] if i > 0 else 0.0)

        # Update State
        prev_x, prev_y = bx, by
        prev_vx, prev_vy = vx, vy
        
        # --- 2. Team Lists & Centroids ---
        t0_pts = [p for p in players if p["team"] == team0]
        t1_pts = [p for p in players if p["team"] == team1]
        
        def get_stats(pts):
            if not pts: return 0.5, 0.5, 0.0, 0.0
            arr = np.array([[p["x"], p["y"]] for p in pts])
            c = arr.mean(axis=0)
            s = arr.std(axis=0)
            return c[0], c[1], s[0], s[1] # cx, cy, std_x, std_y

        t0_cx, t0_cy, t0_sx, t0_sy = get_stats(t0_pts)
        t1_cx, t1_cy, t1_sx, t1_sy = get_stats(t1_pts)

        # --- 3. Distances & Nearest Neighbors ---
        dist_t0_c = ((bx - t0_cx)**2 + (by - t0_cy)**2)**0.5
        dist_t1_c = ((bx - t1_cx)**2 + (by - t1_cy)**2)**0.5
        
        # Nearest logic
        min_dist_any = 999.0
        is_nearest_t0 = 0.0
        min_dist_t0 = 999.0
        min_dist_t1 = 999.0
        
        # Pressure / Clustering
        pressure_idx = 0.0
        close_cnt = 0
        
        # Packing (assuming playing left-to-right or generic X packing)
        # We can calculate "Players behind ball" relative to team centroid direction
        # Simple heuristic: how many opponents have x < ball_x (if team1 is right side)
        # We will just count count(x < bx) and count(x > bx) to let model decide
        pack_t0 = 0
        pack_t1 = 0

        all_valid = t0_pts + t1_pts
        for p in all_valid:
            d = ((bx - p["x"])**2 + (by - p["y"])**2)**0.5
            
            # Global Nearest
            if d < min_dist_any:
                min_dist_any = d
                is_nearest_t0 = 1.0 if p["team"] == team0 else 0.0
            
            # Team Nearest
            if p["team"] == team0:
                if d < min_dist_t0: min_dist_t0 = d
                # Packing: simple check relative to ball X
                if p["x"] < bx: pack_t0 += 1
            else:
                if d < min_dist_t1: min_dist_t1 = d
                if p["x"] > bx: pack_t1 += 1 # Assume other team direction? 
                # Actually, simple "behind ball" is ambiguous without knowing attack direction.
                # Let's just use "players to the left of ball" vs "players to right".
                # Refined:
                if p["x"] < bx: pack_t1 += 1 # Re-using variable for "players on left"

        # Re-calc packing cleanly
        # pack_t0 = count of T0 players left of ball
        # pack_t1 = count of T1 players left of ball
        pack_t0 = sum(1 for p in t0_pts if p["x"] < bx)
        pack_t1 = sum(1 for p in t1_pts if p["x"] < bx)

        # Pressure index (Sum of 1/d for close players)
        for p in all_valid:
            d = ((bx - p["x"])**2 + (by - p["y"])**2)**0.5
            if d < 0.001: d = 0.001 # Clamp
            if d < (10.0 / FIELD_WIDTH_M): # Within ~10m
                pressure_idx += (1.0 / d)
            if d < DIST_CLOSE_THRESH:
                close_cnt += 1

        # --- 4. Fill Vector ---
        feats[i, 0] = bx
        feats[i, 1] = by
        feats[i, 2] = vx
        feats[i, 3] = vy
        feats[i, 4] = speed
        feats[i, 5] = accel
        feats[i, 6] = dist_t0_c
        feats[i, 7] = dist_t1_c
        feats[i, 8] = min_dist_any
        feats[i, 9] = is_nearest_t0
        feats[i, 10] = min_dist_t0
        feats[i, 11] = min_dist_t1
        feats[i, 12] = pressure_idx
        feats[i, 13] = float(close_cnt)
        feats[i, 14] = t0_sx
        feats[i, 15] = t1_sx
        feats[i, 16] = t0_sy
        feats[i, 17] = t1_sy
        feats[i, 18] = abs(0.5 - bx) # Centrality X
        feats[i, 19] = abs(0.5 - by) # Centrality Y
        feats[i, 20] = float(pack_t0)
        feats[i, 21] = float(pack_t1)

    meta = {
        "fps": fps,
        "team_keys": [team0, team1],
        "base_feature_dim": 22,
        "features": [
            "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_speed", "ball_accel",
            "dist_t0_c", "dist_t1_c", "min_dist_any", "is_nearest_t0",
            "min_dist_t0", "min_dist_t1", "pressure_idx", "close_cnt",
            "t0_sx", "t1_sx", "t0_sy", "t1_sy", 
            "centrality_x", "centrality_y", "pack_t0_left", "pack_t1_left"
        ]
    }
    return feats, fps, meta


def build_event_feature_vector(
    center_idx: int,
    base_feats: np.ndarray,
    fps: float,
    window_seconds: float = 0.8, # Increased window slightly
) -> np.ndarray:
    """
    Stacks temporal statistics around the center frame.
    Input: (N, 22) matrix
    Output: Single 1D vector of shape (~88+,)
    """
    n_frames, d = base_feats.shape
    
    # 1. Define Window
    w = max(1, int(round(window_seconds * fps)))
    lo = max(0, center_idx - w)
    hi = min(n_frames, center_idx + w + 1)
    
    window = base_feats[lo:hi] # (Frames, 22)
    center = base_feats[center_idx] # (22,)

    # 2. Statistics
    mean_win = window.mean(axis=0) # (22,)
    std_win = window.std(axis=0)   # (22,)
    
    # 3. Deltas (Start vs End of window) - Captures "Change over event"
    win_start = window[0]
    win_end = window[-1]
    delta_win = win_end - win_start # (22,)

    # 4. Immediate Deltas (Velocity of features)
    prev = base_feats[max(0, center_idx - 2)]
    nxt = base_feats[min(n_frames - 1, center_idx + 2)]
    delta_immediate = nxt - prev # (22,)

    # 5. Concatenate
    # We select specific sub-parts to keep dims reasonable if needed, 
    # but for Random Forest, 100 dims is fine.
    # Structure: Center(22) + Mean(22) + Std(22) + DeltaWin(22) = 88 features
    
    feat_vec = np.concatenate([
        center, 
        mean_win, 
        std_win, 
        delta_win
    ]).astype(np.float32)
    
    return feat_vec
