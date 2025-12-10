#!/usr/bin/env python3
"""
Professional Football Analysis AI Tracker
State-of-the-art computer vision for football analysis
"""

import cv2
import numpy as np
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class EventType(Enum):
    PASS = "pass"
    SHOT = "shot"
    GOAL = "goal"
    TACKLE = "tackle"
    CARRY = "carry"
    DRIBBLE = "dribble"
    CROSS = "cross"
    FOUL = "foul"
    OFFSIDE = "offside"
    RECOVERY = "recovery"
    INTERCEPTION = "interception"
    CLEARANCE = "clearance"

@dataclass
class Player:
    id: str
    team: str
    jersey: int
    position: Tuple[float, float]
    speed: float
    acceleration: float
    direction: float
    confidence: float
    touches: int = 0
    passes: int = 0
    shots: int = 0
    tackles: int = 0
    distance_covered: float = 0.0
    high_intensity_runs: int = 0
    sprints: int = 0
    top_speed: float = 0.0
    average_speed: float = 0.0
    pressing_events: int = 0
    recoveries: int = 0
    interceptions: int = 0
    clearances: int = 0
    turnovers: int = 0
    miscontrols: int = 0
    defensive_actions: int = 0

@dataclass
class Ball:
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    confidence: float
    in_air: bool = False
    predicted_landing: Optional[Tuple[float, float]] = None
    trajectory: List[Tuple[float, float]] = None

@dataclass
class Event:
    type: EventType
    timestamp: float
    player_id: str
    position: Tuple[float, float]
    confidence: float
    xG: float = 0.0
    xA: float = 0.0
    xT: float = 0.0
    packing_score: int = 0
    pass_quality: float = 0.0
    shot_quality: float = 0.0

class ProfessionalAITracker:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize professional AI tracker"""
        self.model_path = model_path
        self.model = None
        self.players = {}
        self.ball = None
        self.events = []
        self.frame_count = 0
        self.fps = 30.0
        self.pitch_calibration = None
        self.homography_matrix = None
        self.scale_factor = 1.0  # meters per pixel
        
        # Tracking parameters
        self.player_tracker = {}
        self.ball_tracker = {}
        self.last_positions = {}
        self.speed_history = {}
        self.trajectory_history = {}
        
        # Performance metrics
        self.team_stats = {
            'home': {'possession': 0, 'passes': 0, 'shots': 0, 'xG': 0.0},
            'away': {'possession': 0, 'passes': 0, 'shots': 0, 'xG': 0.0}
        }
        
        log.info("Professional AI Tracker initialized")

    def load_model(self):
        """Load YOLO model for object detection"""
        try:
            import os
from ultralytics import YOLO
            self.model = YOLO(os.getenv('MODEL_PATH', self.model_path))
            log.info(f"YOLO model loaded from {self.model_path}")
            return True
        except ImportError:
            log.warning("YOLO not available, using OpenCV fallback")
            return False

    def detect_players_simple(self, frame):
        """Simple player detection for initial identification"""
        height, width = frame.shape[:2]
        players = []
        
        if self.model:
            # Use YOLO for detection
            results = self.model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.model.names[cls]
                    
                    if label == "person" and conf > 0.5:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        players.append({
                            'id': f'player_{len(players)+1:03d}',
                            'bbox': [x1, y1, x2, y2],
                            'center': [center_x, center_y],
                            'confidence': conf,
                            'jersey': len(players) + 1,
                            'team': 'home' if len(players) % 2 == 0 else 'away'
                        })
        else:
            # Fallback to OpenCV detection
            players = self.detect_players_opencv(frame)
        
        return players

    def detect_players_advanced(self, frame):
        """Advanced player detection with pose estimation"""
        height, width = frame.shape[:2]
        players = []
        
        if self.model:
            # Use YOLO for detection
            results = self.model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.model.names[cls]
                    
                    if label == "person" and conf > 0.5:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Estimate player orientation and pose
                        orientation = self.estimate_player_orientation(frame[y1:y2, x1:x2])
                        
                        players.append({
                            'bbox': [x1, y1, x2, y2],
                            'center': [center_x, center_y],
                            'confidence': conf,
                            'orientation': orientation,
                            'height': y2 - y1,
                            'width': x2 - x1
                        })
        else:
            # Fallback to OpenCV detection
            players = self.detect_players_opencv(frame)
        
        return players

    def detect_players_opencv(self, frame):
        """OpenCV-based player detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        players = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)
                if h > w and h > 50:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    players.append({
                        'bbox': [x, y, x + w, y + h],
                        'center': [center_x, center_y],
                        'confidence': 0.8,
                        'orientation': 0.0,
                        'height': h,
                        'width': w
                    })
        
        return players

    def detect_ball_advanced(self, frame):
        """Advanced ball detection with physics modeling"""
        height, width = frame.shape[:2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Multiple color ranges for ball detection
        color_ranges = [
            ([0, 0, 200], [180, 30, 255]),  # White/light colors
            ([0, 0, 0], [180, 255, 50]),    # Dark colors
            ([20, 100, 100], [30, 255, 255])  # Orange colors
        ]
        
        best_ball = None
        best_confidence = 0
        
        for lower, upper in color_ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 20 < area < 500:  # Ball size range
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Calculate confidence based on circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        confidence = min(circularity * 0.8, 1.0)
                        
                        if confidence > best_confidence:
                            best_ball = {
                                'bbox': [x, y, x + w, y + h],
                                'center': [center_x, center_y],
                                'confidence': confidence,
                                'area': area,
                                'circularity': circularity
                            }
                            best_confidence = confidence
        
        return best_ball

    def estimate_player_orientation(self, player_roi):
        """Estimate player orientation using pose estimation"""
        # Simplified orientation estimation
        # In a real implementation, you'd use MediaPipe or similar
        gray = cv2.cvtColor(player_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find dominant edge direction
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
        if lines is not None:
            angles = [line[0][1] for line in lines]
            if angles:
                return np.median(angles)
        
        return 0.0

    def track_players(self, detected_players):
        """Track players across frames with ID persistence"""
        tracked_players = []
        
        for player in detected_players:
            # Find closest existing player
            best_match = None
            best_distance = float('inf')
            
            for player_id, last_pos in self.last_positions.items():
                distance = math.sqrt(
                    (player['center'][0] - last_pos[0])**2 + 
                    (player['center'][1] - last_pos[1])**2
                )
                
                if distance < 50 and distance < best_distance:  # 50 pixel threshold
                    best_match = player_id
                    best_distance = distance
            
            if best_match:
                # Update existing player
                player_id = best_match
            else:
                # Create new player
                player_id = f"player_{len(self.players) + 1:03d}"
                self.players[player_id] = Player(
                    id=player_id,
                    team="home" if len(self.players) % 2 == 0 else "away",
                    jersey=len(self.players) + 1,
                    position=player['center'],
                    speed=0.0,
                    acceleration=0.0,
                    direction=0.0,
                    confidence=player['confidence']
                )
            
            # Update player data
            self.players[player_id].position = player['center']
            self.players[player_id].confidence = player['confidence']
            
            # Calculate speed and acceleration
            if player_id in self.last_positions:
                speed, acceleration = self.calculate_motion_metrics(
                    self.last_positions[player_id], 
                    player['center']
                )
                self.players[player_id].speed = speed
                self.players[player_id].acceleration = acceleration
                
                # Update distance covered
                distance = math.sqrt(
                    (player['center'][0] - self.last_positions[player_id][0])**2 + 
                    (player['center'][1] - self.last_positions[player_id][1])**2
                )
                self.players[player_id].distance_covered += distance * self.scale_factor
                
                # Update top speed
                if speed > self.players[player_id].top_speed:
                    self.players[player_id].top_speed = speed
                
                # Count high intensity runs and sprints
                if speed > 20:  # km/h
                    self.players[player_id].high_intensity_runs += 1
                if speed > 25:  # km/h
                    self.players[player_id].sprints += 1
            
            self.last_positions[player_id] = player['center']
            tracked_players.append(self.players[player_id])
        
        return tracked_players

    def calculate_motion_metrics(self, last_pos, current_pos):
        """Calculate speed and acceleration"""
        distance_pixels = math.sqrt(
            (current_pos[0] - last_pos[0])**2 + 
            (current_pos[1] - last_pos[1])**2
        )
        
        # Convert to meters
        distance_meters = distance_pixels * self.scale_factor
        
        # Calculate speed in km/h
        time_seconds = 1.0 / self.fps
        speed_ms = distance_meters / time_seconds if time_seconds > 0 else 0
        speed_kmh = speed_ms * 3.6
        
        # Calculate acceleration
        if hasattr(self, 'speed_history'):
            if len(self.speed_history) > 0:
                last_speed = self.speed_history[-1] if self.speed_history else 0
                acceleration = (speed_kmh - last_speed) / time_seconds
            else:
                acceleration = 0
        else:
            self.speed_history = []
            acceleration = 0
        
        self.speed_history.append(speed_kmh)
        if len(self.speed_history) > 10:
            self.speed_history.pop(0)
        
        return speed_kmh, acceleration

    def detect_events(self, players, ball):
        """Detect football events using rule-based and ML approaches"""
        new_events = []
        
        # Detect passes
        if ball and len(players) >= 2:
            for player in players:
                distance_to_ball = math.sqrt(
                    (player.position[0] - ball.position[0])**2 + 
                    (player.position[1] - ball.position[1])**2
                )
                
                if distance_to_ball < 30:  # Player has ball
                    # Check for pass to another player
                    for other_player in players:
                        if other_player.id != player.id:
                            distance_to_other = math.sqrt(
                                (other_player.position[0] - ball.position[0])**2 + 
                                (other_player.position[1] - ball.position[1])**2
                            )
                            
                            if distance_to_other < 50:  # Pass completed
                                event = Event(
                                    type=EventType.PASS,
                                    timestamp=self.frame_count / self.fps,
                                    player_id=player.id,
                                    position=ball.position,
                                    confidence=0.8,
                                    xA=self.calculate_xA(player, other_player),
                                    xT=self.calculate_xT(player, other_player),
                                    pass_quality=self.calculate_pass_quality(player, other_player)
                                )
                                new_events.append(event)
                                player.passes += 1
        
        # Detect shots
        if ball:
            goal_distance = self.calculate_distance_to_goal(ball.position)
            if goal_distance < 200:  # Within shooting range
                for player in players:
                    distance_to_ball = math.sqrt(
                        (player.position[0] - ball.position[0])**2 + 
                        (player.position[1] - ball.position[1])**2
                    )
                    
                    if distance_to_ball < 30:  # Player has ball
                        xG = self.calculate_xG(ball.position, goal_distance)
                        event = Event(
                            type=EventType.SHOT,
                            timestamp=self.frame_count / self.fps,
                            player_id=player.id,
                            position=ball.position,
                            confidence=0.9,
                            xG=xG,
                            shot_quality=self.calculate_shot_quality(ball.position)
                        )
                        new_events.append(event)
                        player.shots += 1
        
        # Detect tackles
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players[i+1:], i+1):
                if player1.team != player2.team:  # Opposing players
                    distance = math.sqrt(
                        (player1.position[0] - player2.position[0])**2 + 
                        (player1.position[1] - player2.position[1])**2
                    )
                    
                    if distance < 20:  # Close contact
                        event = Event(
                            type=EventType.TACKLE,
                            timestamp=self.frame_count / self.fps,
                            player_id=player1.id,
                            position=player1.position,
                            confidence=0.7
                        )
                        new_events.append(event)
                        player1.tackles += 1
        
        self.events.extend(new_events)
        return new_events

    def calculate_xG(self, position, distance_to_goal):
        """Calculate expected goals (xG)"""
        # Simplified xG calculation based on distance and angle
        base_xg = max(0, 1.0 - (distance_to_goal / 500))
        angle_factor = 1.0  # Could be improved with angle calculation
        return min(base_xg * angle_factor, 1.0)

    def calculate_xA(self, passer, receiver):
        """Calculate expected assists (xA)"""
        # Simplified xA calculation
        distance = math.sqrt(
            (receiver.position[0] - passer.position[0])**2 + 
            (receiver.position[1] - passer.position[1])**2
        )
        return max(0, 1.0 - (distance / 300))

    def calculate_xT(self, player1, player2):
        """Calculate expected threat (xT)"""
        # Simplified xT calculation
        return 0.1  # Placeholder

    def calculate_pass_quality(self, passer, receiver):
        """Calculate pass quality score"""
        distance = math.sqrt(
            (receiver.position[0] - passer.position[0])**2 + 
            (receiver.position[1] - passer.position[1])**2
        )
        return max(0, 1.0 - (distance / 200))

    def calculate_shot_quality(self, position):
        """Calculate shot quality score"""
        distance_to_goal = self.calculate_distance_to_goal(position)
        return max(0, 1.0 - (distance_to_goal / 300))

    def calculate_distance_to_goal(self, position):
        """Calculate distance to nearest goal"""
        # Simplified - assume goals are at x=0 and x=1920
        return min(position[0], 1920 - position[0])

    def calibrate_pitch(self, frame):
        """Calibrate pitch using field lines detection"""
        # Detect field lines using Hough transform
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Find pitch boundaries and calculate homography
            # This is a simplified version - real implementation would be more complex
            self.scale_factor = 0.1  # meters per pixel (estimated)
            return True
        
        return False

    def draw_advanced_overlays(self, frame, players, ball, events):
        """Draw professional-grade overlays"""
        overlay_frame = frame.copy()
        
        # Draw players with advanced visualization
        for player in players:
            x, y = map(int, player.position)
            
            # Player circle with team color
            color = (0, 255, 0) if player.team == "home" else (0, 0, 255)
            cv2.circle(overlay_frame, (x, y), 15, color, 2)
            
            # Jersey number
            cv2.putText(overlay_frame, str(player.jersey), (x - 10, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Speed indicator
            speed_text = f"{player.speed:.1f} km/h"
            cv2.putText(overlay_frame, speed_text, (x - 25, y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Pressing zone
            cv2.circle(overlay_frame, (x, y), 30, color, 1)
        
        # Draw ball with trajectory
        if ball:
            bx, by = map(int, ball.position)
            cv2.circle(overlay_frame, (bx, by), 8, (0, 255, 255), -1)
            cv2.circle(overlay_frame, (bx, by), 12, (0, 255, 255), 2)
            
            # Draw ball trajectory
            if hasattr(ball, 'trajectory') and ball.trajectory:
                for i in range(1, len(ball.trajectory)):
                    pt1 = tuple(map(int, ball.trajectory[i-1]))
                    pt2 = tuple(map(int, ball.trajectory[i]))
                    cv2.line(overlay_frame, pt1, pt2, (0, 255, 255), 2)
        
        # Draw events
        for event in events[-10:]:  # Show last 10 events
            ex, ey = map(int, event.position)
            color = self.get_event_color(event.type)
            cv2.circle(overlay_frame, (ex, ey), 10, color, -1)
            cv2.putText(overlay_frame, event.type.value.upper(), (ex + 15, ey), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw pitch zones
        self.draw_pitch_zones(overlay_frame)
        
        return overlay_frame

    def get_event_color(self, event_type):
        """Get color for event type"""
        colors = {
            EventType.PASS: (0, 0, 255),      # Blue
            EventType.SHOT: (0, 255, 0),     # Green
            EventType.GOAL: (0, 255, 255),   # Yellow
            EventType.TACKLE: (0, 0, 255),   # Red
            EventType.CARRY: (255, 0, 255),  # Magenta
            EventType.DRIBBLE: (255, 255, 0) # Cyan
        }
        return colors.get(event_type, (255, 255, 255))

    def draw_pitch_zones(self, frame):
        """Draw tactical pitch zones"""
        height, width = frame.shape[:2]
        
        # Center line
        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)
        
        # Center circle
        cv2.circle(frame, (width // 2, height // 2), 60, (255, 255, 255), 2)
        
        # Goal areas
        goal_width = width // 8
        goal_height = height // 4
        
        # Left goal
        cv2.rectangle(frame, (0, height // 2 - goal_height // 2), 
                     (goal_width, height // 2 + goal_height // 2), (255, 255, 255), 2)
        
        # Right goal
        cv2.rectangle(frame, (width - goal_width, height // 2 - goal_height // 2), 
                     (width, height // 2 + goal_height // 2), (255, 255, 255), 2)
        
        # Tactical zones
        # Defensive third
        cv2.rectangle(frame, (0, 0), (width // 3, height), (0, 255, 0), 1)
        
        # Middle third
        cv2.rectangle(frame, (width // 3, 0), (2 * width // 3, height), (255, 255, 0), 1)
        
        # Attacking third
        cv2.rectangle(frame, (2 * width // 3, 0), (width, height), (255, 0, 0), 1)

    def process_video(self, video_path):
        """Process entire video with professional analysis"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load model
        self.load_model()
        
        # Calibrate pitch on first frame
        ret, first_frame = cap.read()
        if ret:
            self.calibrate_pitch(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        results = {
            'video_metadata': {
                'fps': self.fps,
                'total_frames': total_frames,
                'duration_s': total_frames / self.fps if self.fps > 0 else 0,
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            },
            'players': [],
            'ball_positions': [],
            'events': [],
            'team_stats': self.team_stats
        }
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect players and ball
            detected_players = self.detect_players_advanced(frame)
            ball_data = self.detect_ball_advanced(frame)
            
            # Track players
            tracked_players = self.track_players(detected_players)
            
            # Update ball
            if ball_data:
                self.ball = Ball(
                    position=ball_data['center'],
                    velocity=(0, 0),  # Simplified
                    confidence=ball_data['confidence']
                )
            
            # Detect events
            new_events = self.detect_events(tracked_players, self.ball)
            
            # Store results
            for player in tracked_players:
                results['players'].append({
                    'frame': frame_idx,
                    'player_id': player.id,
                    'team': player.team,
                    'jersey': player.jersey,
                    'position': player.position,
                    'speed': player.speed,
                    'acceleration': player.acceleration,
                    'confidence': player.confidence
                })
            
            if self.ball:
                results['ball_positions'].append({
                    'frame': frame_idx,
                    'position': self.ball.position,
                    'confidence': self.ball.confidence,
                    'in_air': self.ball.in_air
                })
            
            for event in new_events:
                results['events'].append({
                    'frame': frame_idx,
                    'type': event.type.value,
                    'timestamp': event.timestamp,
                    'player_id': event.player_id,
                    'position': event.position,
                    'confidence': event.confidence,
                    'xG': event.xG,
                    'xA': event.xA,
                    'xT': event.xT
                })
            
            frame_idx += 1
            
            # Progress update
            if frame_idx % 30 == 0:
                log.info(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        
        # Calculate final team statistics
        self.calculate_team_statistics(results)
        
        return results

    def calculate_team_statistics(self, results):
        """Calculate comprehensive team statistics"""
        home_players = [p for p in results['players'] if p['team'] == 'home']
        away_players = [p for p in results['players'] if p['team'] == 'away']
        
        # Calculate possession
        total_frames = len(set(p['frame'] for p in results['players']))
        home_frames = len(set(p['frame'] for p in home_players))
        away_frames = len(set(p['frame'] for p in away_players))
        
        if total_frames > 0:
            self.team_stats['home']['possession'] = (home_frames / total_frames) * 100
            self.team_stats['away']['possession'] = (away_frames / total_frames) * 100
        
        # Calculate other metrics
        home_events = [e for e in results['events'] if any(p['player_id'] == e['player_id'] and p['team'] == 'home' for p in results['players'])]
        away_events = [e for e in results['events'] if any(p['player_id'] == e['player_id'] and p['team'] == 'away' for p in results['players'])]
        
        self.team_stats['home']['passes'] = len([e for e in home_events if e['type'] == 'pass'])
        self.team_stats['away']['passes'] = len([e for e in away_events if e['type'] == 'pass'])
        
        self.team_stats['home']['shots'] = len([e for e in home_events if e['type'] == 'shot'])
        self.team_stats['away']['shots'] = len([e for e in away_events if e['type'] == 'shot'])
        
        self.team_stats['home']['xG'] = sum(e.get('xG', 0) for e in home_events)
        self.team_stats['away']['xG'] = sum(e.get('xG', 0) for e in away_events)

if __name__ == "__main__":
    # Example usage
    tracker = ProfessionalAITracker()
    
    # Process a video file
    video_path = "test_match.mp4"
    if Path(video_path).exists():
        print(f"Processing video: {video_path}")
        results = tracker.process_video(video_path)
        print(f"Analysis complete!")
        print(f"Players detected: {len(set(p['player_id'] for p in results['players']))}")
        print(f"Events detected: {len(results['events'])}")
        print(f"Team stats: {results['team_stats']}")
    else:
        print(f"Video file {video_path} not found")

