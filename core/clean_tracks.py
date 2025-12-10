#!/usr/bin/env python
import json
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_FILE = "runs/json/formatted_tracks_silver.json"
OUTPUT_FILE = "runs/json/formatted_tracks_silver.json"

# --- AGGRESSIVE CLEANING CONFIG ---
MAX_GAP_FILL = 25      # 1 second gap fill (Stabilizes IDs)
MIN_TRACK_LEN = 30     # 1.2 second minimum (Removes Ghosts)
BOTTOM_CROP = 0.85     # CUTS THE BOTTOM 15% (Removes Coaches)

def load_tracks():
    if not Path(INPUT_FILE).exists(): return None, 0
    data = json.loads(Path(INPUT_FILE).read_text())
    return data.get("frames", []), data.get("fps", 25.0)

def main():
    print("🧹 Starting AGGRESSIVE Track Cleaning...")
    frames, fps = load_tracks()
    if not frames: return

    rows = []
    for f in frames:
        t = f["t"]
        for p in f["players"]:
            rows.append({"frame": int(t * fps), "id": str(p["id"]), "x": p["x"], "y": p["y"], "team": p.get("team", "unknown"), "type": "player"})
        if f["ball"]:
            b = f["ball"]
            rows.append({"frame": int(t * fps), "id": "ball", "x": b["x"], "y": b["y"], "team": "ball", "type": "ball"})

    df = pd.DataFrame(rows)
    if df.empty: return

    print(f"Loaded {len(df)} points.")
    
    # SPATIAL FILTER (The "Kill Zone")
    max_y = df["y"].max()
    limit = (1080 * BOTTOM_CROP) if max_y > 2.0 else BOTTOM_CROP
    print(f"   Filtering Y > {limit:.1f} (removing coaches)...")
    
    initial_count = len(df)
    # Keep Ball OR Players ABOVE the line
    df = df[ (df["type"] == "ball") | (df["y"] < limit) ]
    print(f"   Removed {initial_count - len(df)} points in the Exclusion Zone.")

    cleaned_rows = []
    for uid in df["id"].unique():
        sub = df[df["id"] == uid].drop_duplicates(subset=["frame"]).sort_values("frame").set_index("frame")
        
        # Check Length (Ghost busting)
        if uid != "ball" and len(sub) < MIN_TRACK_LEN: continue

        # Interpolate (Glue)
        full_idx = range(sub.index.min(), sub.index.max() + 1)
        sub = sub.reindex(full_idx)
        sub["x"] = sub["x"].interpolate(limit=MAX_GAP_FILL)
        sub["y"] = sub["y"].interpolate(limit=MAX_GAP_FILL)
        sub["id"] = uid
        sub["type"] = "ball" if uid == "ball" else "player"
        sub["team"] = sub["team"].ffill().bfill()
        
        cleaned_rows.append(sub.dropna(subset=["x"]))

    if not cleaned_rows: return
    df_clean = pd.concat(cleaned_rows).reset_index()
    
    # Rebuild JSON
    output_frames = []
    grouped = df_clean.groupby("frame")
    for f_idx in range(int(df_clean["frame"].min()), int(df_clean["frame"].max()) + 1):
        g = grouped.get_group(f_idx) if f_idx in grouped.groups else pd.DataFrame()
        
        ball_obj = None
        ball_row = g[g["type"] == "ball"]
        if not ball_row.empty: ball_obj = {"x": ball_row.iloc[0]["x"], "y": ball_row.iloc[0]["y"]}
        
        players = []
        for _, r in g[g["type"] == "player"].iterrows():
            players.append({"id": r["id"], "x": r["x"], "y": r["y"], "team": r["team"]})
            
        output_frames.append({"t": float(f_idx / fps), "ball": ball_obj, "players": players})

    with open(OUTPUT_FILE, "w") as f: json.dump({"fps": fps, "frames": output_frames}, f)
    print(f"✨ Cleaned tracks saved.")

if __name__ == "__main__":
    main()
