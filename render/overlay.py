# render/overlay.py
# (Kept minimal; main overlay is already done inside tracking loop.
#  This file is a hook if you later want to re-render from tracks.json.)

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def render_overlay_from_tracks(
    video_path: str,
    tracks_json_path: str,
    output_path: str,
    draw_trails: int = 30
) -> str:
    """
    Re-render overlay from tracks.json file
    """
    # Load tracks
    with open(tracks_json_path, 'r') as f:
        tracks_data = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    # Process frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw player tracks for this frame
        for player_data in tracks_data.get("players", []):
            tid = player_data["tid"]
            frames = player_data["frames"]
            
            # Find current frame data
            current_frame_data = None
            for f_data in frames:
                if f_data["f"] == frame_idx:
                    current_frame_data = f_data
                    break
            
            if current_frame_data:
                cx = int(current_frame_data["cx"] * W)
                cy = int(current_frame_data["cy"] * H)
                w = int(current_frame_data["w"] * W)
                h = int(current_frame_data["h"] * H)
                
                # Draw player
                cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 2)
                cv2.putText(frame, str(tid), (cx - 10, cy + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw ball for this frame
        for ball_data in tracks_data.get("ball", []):
            if ball_data["f"] == frame_idx:
                bx = int(ball_data["cx"] * W)
                by = int(ball_data["cy"] * H)
                cv2.circle(frame, (bx, by), 7, (0, 255, 255), -1)
                break
        
        writer.write(frame)
        frame_idx += 1
    
    cap.release()
    writer.release()
    
    return output_path
