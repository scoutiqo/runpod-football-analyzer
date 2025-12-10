#!/usr/bin/env python
import argparse
import json
import cv2
import sys
import os
import numpy as np
from tqdm import tqdm

def load_tracks(path):
    with open(path) as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "frames" in data:
        return data["frames"]
    return []

def run(video_path, tracks_path, out_path):
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    print(f"Loading tracks from {tracks_path}...")
    frames_data = load_tracks(tracks_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[overlay] Video FPS={fps}, frames={total}, {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    COLOR_BALL = (0, 255, 255)
    COLOR_A = (0, 0, 255)
    COLOR_B = (255, 0, 0)
    COLOR_UNK = (200, 200, 200)

    for i in tqdm(range(total)):
        ret, frame = cap.read()
        if not ret: break
        
        if i < len(frames_data):
            data = frames_data[i]
            
            # Players
            players = data.get("players", []) or data.get("objects", [])
            for p in players:
                # Coordinates
                cx, cy = 0, 0
                if "x" in p and p["x"] <= 1.0:
                    cx, cy = int(p["x"] * width), int(p["y"] * height)
                elif "x" in p:
                    cx, cy = int(p["x"]), int(p["y"])
                elif "bbox" in p:
                    cx = int((p["bbox"][0] + p["bbox"][2])/2)
                    cy = int((p["bbox"][1] + p["bbox"][3])/2)
                else:
                    continue
                
                team = str(p.get("team", "unknown"))
                color = COLOR_UNK
                if team in ["0", "A", "Team A"]: color = COLOR_A
                if team in ["1", "B", "Team B"]: color = COLOR_B
                
                # Draw Circle
                cv2.circle(frame, (cx, cy), 10, color, 2)
                
                # Draw ID
                pid = str(p.get("id",""))
                cv2.putText(frame, pid, (cx-5, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # --- DRAW SPEED (New) ---
                speed = p.get("speed", 0)
                if speed > 3.0: # Only show if running > 3km/h
                    label = f"{speed} km/h"
                    # Draw background box
                    (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(frame, (cx-20, cy+15), (cx-20+w_text, cy+15-h_text-5), (0,0,0), -1)
                    # Draw text
                    cv2.putText(frame, label, (cx-20, cy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            # Ball
            ball = data.get("ball")
            if ball:
                bx, by = 0, 0
                if "x" in ball and ball["x"] <= 1.0:
                    bx, by = int(ball["x"] * width), int(ball["y"] * height)
                elif "x" in ball:
                    bx, by = int(ball["x"]), int(ball["y"])
                elif "x_px" in ball:
                    bx, by = int(ball["x_px"]), int(ball["y_px"])
                
                if bx > 0:
                    cv2.circle(frame, (bx, by), 8, COLOR_BALL, -1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved overlay video to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.video, args.tracks, args.out)
