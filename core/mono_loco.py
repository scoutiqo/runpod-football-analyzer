import numpy as np
from ultralytics import YOLO
import cv2
import os

class MonoLocoEstimator:
    def __init__(self, focal_length=1200):
        print("   🧠 Loading YOLOv8-Pose for 3D Skeleton Tracking...")
        self.model = YOLO('yolov8n-pose.pt')
        self.AVG_PLAYER_HEIGHT = 1.75 
        self.FOCAL_LENGTH = focal_length 
        
    def predict(self, frame):
        # Run Inference
        results = self.model(frame, verbose=False, classes=[0])[0]
        players_3d = []
        
        # Safety Check: Ensure both keypoints and boxes exist
        if results.keypoints is None or results.boxes is None:
            return []

        # Extract Data
        # keypoints.data is (N, 17, 3)
        keypoints = results.keypoints.data.cpu().numpy()
        # boxes.xyxy is (N, 4)
        boxes = results.boxes.xyxy.cpu().numpy()
        
        # Verify Shapes Match
        if len(keypoints) == 0 or len(boxes) == 0 or len(keypoints) != len(boxes):
            return []

        # Get IDs if available (ByteTrack inside YOLO)
        ids = results.boxes.id
        track_ids = ids.cpu().numpy() if ids is not None else [-1] * len(boxes)
            
        for i, box in enumerate(boxes):
            # Ensure index is valid (redundant but safe)
            if i >= len(track_ids): break

            # Calculate Height (y2 - y1)
            box_h = box[3] - box[1]
            
            # Filter Noise: Too small to be a player
            if box_h < 10: continue 

            # 3D Math (Pinhole Camera Model)
            depth_m = (self.FOCAL_LENGTH * self.AVG_PLAYER_HEIGHT) / box_h
            
            img_cx = frame.shape[1] / 2
            box_cx = (box[0] + box[2]) / 2
            
            # Lateral Position (Meters)
            x_m = (box_cx - img_cx) * depth_m / self.FOCAL_LENGTH
            
            players_3d.append({
                "id": int(track_ids[i]),
                "box": box,
                "position_3d": [float(x_m), 0.0, float(depth_m)],
                "height_px": float(box_h)
            })
            
        return players_3d
