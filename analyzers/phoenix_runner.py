# analyzers/phoenix_runner.py
import numpy as np
import cv2
import torch
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# Import our Phoenix components
from phoenix_opt.window_opt import SlidingWindowOptimizer, optimize_window
from phoenix_opt.confnet import ConfidenceManager
from phoenix_opt.field_detector import FieldLineDetector

logger = logging.getLogger(__name__)

class PhoenixRunner:
    """
    Main runner for adaptive monocular 2.5D tracking using sliding window optimization
    """
    
    def __init__(self, 
                 window_size=25,  # ~1 second at 25fps
                 overlap=5,
                 device="cpu"):
        self.window_size = window_size
        self.overlap = overlap
        self.device = device
        
        # Initialize components
        self.optimizer = SlidingWindowOptimizer(window_size, overlap, device)
        self.confidence_manager = ConfidenceManager(device)
        self.field_detector = FieldLineDetector(device=device)
        
        # State tracking
        self.current_frame = 0
        self.measurement_buffer = []
        self.confidence_buffer = []
        self.field_buffer = []
        
        # Results storage
        self.optimized_tracks = []
        self.camera_poses = []
        
        # Camera intrinsics (will be estimated)
        self.K = None
        
    def set_camera_intrinsics(self, K):
        """Set camera intrinsics matrix"""
        self.K = K
        self.optimizer.set_camera_intrinsics(K)
    
    def process_frame(self, frame, detections):
        """
        Process a single frame and return optimized results
        
        Args:
            frame: (H, W, 3) BGR image
            detections: dict with player and ball detections
        
        Returns:
            dict with optimized world coordinates and confidence scores
        """
        self.current_frame += 1
        
        # Extract measurements from detections
        measurements = self._extract_measurements(detections)
        
        # Detect field lines and keypoints
        field_result = self.field_detector.process_frame(frame)
        
        # Predict confidences using learned networks
        confidences = self._predict_confidences(measurements, frame)
        
        # Add to buffers
        self.measurement_buffer.append(measurements)
        self.confidence_buffer.append(confidences)
        self.field_buffer.append(field_result)
        
        # Keep only recent frames
        if len(self.measurement_buffer) > self.window_size:
            self.measurement_buffer.pop(0)
            self.confidence_buffer.pop(0)
            self.field_buffer.pop(0)
        
        # Run optimization if we have enough frames
        if len(self.measurement_buffer) >= self.window_size:
            return self._optimize_window()
        else:
            # Return partial results
            return self._get_partial_results()
    
    def _extract_measurements(self, detections):
        """Extract measurements from detections"""
        measurements = {
            "u_players": [],
            "u_ball": None,
            "h_ball": None,
            "bbox_players": [],
            "bbox_ball": None
        }
        
        # Player measurements
        players = detections.get("players", [])
        for player in players:
            bbox = player.get("bbox", [0, 0, 0, 0])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                # Use bottom center of bbox (ground contact point)
                u_center = (x1 + x2) / 2
                v_bottom = y2
                measurements["u_players"].append([u_center, v_bottom])
                measurements["bbox_players"].append(bbox)
        
        # Ball measurements
        ball = detections.get("ball")
        if ball and ball.get("bbox"):
            bbox = ball["bbox"]
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                u_center = (x1 + x2) / 2
                v_center = (y1 + y2) / 2
                h_size = max(x2 - x1, y2 - y1)  # Ball size in pixels
                
                measurements["u_ball"] = [u_center, v_center]
                measurements["h_ball"] = h_size
                measurements["bbox_ball"] = bbox
        
        return measurements
    
    def _predict_confidences(self, measurements, frame):
        """Predict confidences for all measurements"""
        confidences = {
            "u_players_sigma": [],
            "u_ball_sigma": None,
            "h_ball_sigma": None,
            "field_sigma": None
        }
        
        # Player confidences
        players = measurements.get("bbox_players", [])
        if players:
            player_detections = []
            for i, bbox in enumerate(players):
                detection = {
                    "bbox": bbox,
                    "conf": 0.8,  # Default confidence
                    "motion": {
                        "speed": 0.0,
                        "direction_change": 0.0,
                        "acceleration": 0.0,
                        "prediction_error": 0.0
                    }
                }
                player_detections.append(detection)
            
            player_confidences = self.confidence_manager.predict_player_confidence(
                player_detections, frame
            )
            confidences["u_players_sigma"] = player_confidences
        
        # Ball confidences
        if measurements["u_ball"] is not None:
            ball_detection = {
                "bbox": measurements["bbox_ball"],
                "conf": 0.8,
                "motion": {
                    "speed": 0.0,
                    "direction_change": 0.0,
                    "acceleration": 0.0,
                    "prediction_error": 0.0
                }
            }
            ball_confidences = self.confidence_manager.predict_ball_confidence(
                [ball_detection], frame
            )
            confidences["u_ball_sigma"] = ball_confidences[0]
            confidences["h_ball_sigma"] = ball_confidences[0]
        
        # Field confidences
        if len(self.field_buffer) > 0:
            field_result = self.field_buffer[-1]
            field_detection = {
                "lines": [],  # Would be populated from field_result
                "intersections": field_result["keypoints"],
                "geometric_consistency": field_result["confidence"]
            }
            field_confidences = self.confidence_manager.predict_field_confidence(
                [field_detection], frame
            )
            confidences["field_sigma"] = field_confidences[0]
        
        return confidences
    
    def _optimize_window(self):
        """Run sliding window optimization"""
        try:
            # Prepare data for optimization
            measurements = self._prepare_measurements()
            confidences = self._prepare_confidences()
            pitch_keypoints = self._prepare_pitch_keypoints()
            
            # Run optimization
            result = self.optimizer.optimize_window(
                measurements, confidences, pitch_keypoints
            )
            
            # Store results
            self.optimized_tracks.append(result)
            if len(self.optimized_tracks) > 10:  # Keep only recent results
                self.optimized_tracks.pop(0)
            
            # Convert to output format
            return self._format_results(result)
            
        except Exception as e:
            logger.error(f"Window optimization failed: {e}")
            return self._get_partial_results()
    
    def _prepare_measurements(self):
        """Prepare measurements for optimization"""
        measurements = {
            "u_players": [],
            "u_ball": [],
            "h_ball": []
        }
        
        for frame_meas in self.measurement_buffer:
            # Players
            if frame_meas["u_players"]:
                measurements["u_players"].append(frame_meas["u_players"])
            else:
                measurements["u_players"].append([])
            
            # Ball
            if frame_meas["u_ball"] is not None:
                measurements["u_ball"].append(frame_meas["u_ball"])
                measurements["h_ball"].append(frame_meas["h_ball"])
            else:
                measurements["u_ball"].append(None)
                measurements["h_ball"].append(None)
        
        # Convert to numpy arrays
        if measurements["u_players"]:
            max_players = max(len(players) for players in measurements["u_players"])
            u_players_array = np.zeros((len(measurements["u_players"]), max_players, 2))
            for i, players in enumerate(measurements["u_players"]):
                for j, player in enumerate(players):
                    if j < max_players:
                        u_players_array[i, j] = player
            measurements["u_players"] = u_players_array
        else:
            measurements["u_players"] = np.zeros((len(self.measurement_buffer), 0, 2))
        
        return measurements
    
    def _prepare_confidences(self):
        """Prepare confidences for optimization"""
        confidences = {
            "u_players_sigma": [],
            "u_ball_sigma": [],
            "h_ball_sigma": [],
            "field_sigma": []
        }
        
        for frame_conf in self.confidence_buffer:
            confidences["u_players_sigma"].append(frame_conf["u_players_sigma"])
            confidences["u_ball_sigma"].append(frame_conf["u_ball_sigma"])
            confidences["h_ball_sigma"].append(frame_conf["h_ball_sigma"])
            confidences["field_sigma"].append(frame_conf["field_sigma"])
        
        return confidences
    
    def _prepare_pitch_keypoints(self):
        """Prepare pitch keypoints for optimization"""
        pitch_keypoints = []
        
        for field_result in self.field_buffer:
            if field_result["image_points"]:
                pitch_keypoints.append(field_result["image_points"])
            else:
                pitch_keypoints.append(None)
        
        return pitch_keypoints
    
    def _format_results(self, result):
        """Format optimization results for output"""
        # Get the last frame results (most recent)
        last_frame_idx = -1
        
        players = []
        if result["players_xy"].shape[1] > 0:
            for i in range(result["players_xy"].shape[1]):
                player_data = {
                    "id": i,
                    "position_world": result["players_xy"][last_frame_idx, i].tolist(),
                    "position_px": [0, 0],  # Would need to project back
                    "team": "unknown",
                    "bbox": [0, 0, 0, 0],
                    "confidence_scores": {"combined": 0.8},
                    "visibility_score": 0.8
                }
                players.append(player_data)
        
        ball_data = None
        if result["ball_xyz"][last_frame_idx] is not None:
            ball_pos = result["ball_xyz"][last_frame_idx]
            ball_vel = result["ball_v"][last_frame_idx]
            
            ball_data = {
                "position_world": ball_pos.tolist(),
                "velocity_world": ball_vel.tolist(),
                "speed_mps": np.linalg.norm(ball_vel),
                "speed_kmh": np.linalg.norm(ball_vel) * 3.6,
                "height": ball_pos[2],
                "in_flight": ball_pos[2] > 0.1,
                "confidence": 0.8,
                "confidence_scores": {"combined": 0.8},
                "visibility_score": 0.8
            }
        
        return {
            "frame": self.current_frame,
            "timestamp": self.current_frame / 30.0,
            "players": players,
            "ball": ball_data,
            "homography_available": True,
            "pose_confidence": 0.8,
            "adaptive_tracking": True,
            "phoenix_optimized": True,
            "metrics": {
                "players_count": len(players),
                "ball_detected": ball_data is not None,
                "frame_rate": 30.0,
                "progress": 0.0,
                "homography_available": True,
                "total_distance_km": 0.0,
                "max_speed_kmh": 0.0,
                "total_sprints": 0
            }
        }
    
    def _get_partial_results(self):
        """Get partial results when not enough frames for optimization"""
        return {
            "frame": self.current_frame,
            "timestamp": self.current_frame / 30.0,
            "players": [],
            "ball": None,
            "homography_available": False,
            "pose_confidence": 0.0,
            "adaptive_tracking": True,
            "phoenix_optimized": False,
            "metrics": {
                "players_count": 0,
                "ball_detected": False,
                "frame_rate": 30.0,
                "progress": 0.0,
                "homography_available": False,
                "total_distance_km": 0.0,
                "max_speed_kmh": 0.0,
                "total_sprints": 0
            }
        }

def run_adaptive_monocular(video_path: str, outdir: str, K: np.ndarray = None):
    """
    Main function to run adaptive monocular tracking on a video
    
    Args:
        video_path: path to input video
        outdir: output directory for results
        K: camera intrinsics matrix (optional)
    
    Returns:
        dict with tracking results
    """
    logger.info(f"Starting adaptive monocular tracking on {video_path}")
    
    # Create output directory
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    # Initialize Phoenix runner
    runner = PhoenixRunner(device="cpu")
    
    # Set camera intrinsics
    if K is not None:
        runner.set_camera_intrinsics(K)
    else:
        # Default camera intrinsics (will be estimated)
        K_default = np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ], dtype=np.float32)
        runner.set_camera_intrinsics(K_default)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {total_frames} frames at {fps} FPS")
    
    # Process video
    results = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Mock detections (would come from YOLO/ByteTrack)
            detections = {
                "players": [
                    {
                        "id": i,
                        "bbox": [100 + i*50, 200 + i*20, 120 + i*50, 250 + i*20],
                        "conf": 0.8
                    }
                    for i in range(22)  # 22 players
                ],
                "ball": {
                    "bbox": [300, 250, 320, 270],
                    "conf": 0.7
                }
            }
            
            # Process frame
            result = runner.process_frame(frame, detections)
            results.append(result)
            
            # Log progress
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
    
    finally:
        cap.release()
    
    # Save results
    output_file = Path(outdir) / "tracks_phoenix.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_file}")
    
    return {
        "results": results,
        "output_file": str(output_file),
        "total_frames": frame_count,
        "fps": fps
    }

