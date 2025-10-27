# metrics/physical.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)

class SpeedTracker:
    """Track player speed, distance, and sprint metrics"""
    
    def __init__(self, player_id: int, smoothing_window: int = 5):
        self.player_id = player_id
        self.smoothing_window = smoothing_window
        
        # Position history for smoothing
        self.position_history = deque(maxlen=smoothing_window)
        self.time_history = deque(maxlen=smoothing_window)
        
        # Metrics
        self.dist_total = 0.0
        self.max_speed = 0.0
        self.avg_speed = 0.0
        self.sprint_windows = []  # list of (t_start, t_end)
        self.sprint_distance = 0.0
        
        # State tracking
        self._sprint_active = False
        self._sprint_start = None
        self._sprint_start_pos = None
        
        # Speed thresholds
        self.sprint_threshold = 7.0  # m/s ~ 25.2 km/h
        self.walk_threshold = 2.0    # m/s ~ 7.2 km/h
        self.run_threshold = 4.0     # m/s ~ 14.4 km/h
        
        # Activity zones
        self.zone_distances = {
            'walking': 0.0,
            'jogging': 0.0, 
            'running': 0.0,
            'sprinting': 0.0
        }
        
    def update(self, t_sec: float, x_m: float, y_m: float):
        """Update tracker with new position"""
        # Add to history
        self.position_history.append((x_m, y_m))
        self.time_history.append(t_sec)
        
        # Need at least 2 points to calculate speed
        if len(self.position_history) < 2:
            return
        
        # Calculate instantaneous speed
        current_pos = (x_m, y_m)
        prev_pos = self.position_history[-2]
        dt = t_sec - self.time_history[-2]
        
        if dt <= 0:
            return
        
        # Distance moved
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Instantaneous speed
        speed = distance / dt
        
        # Add to total distance
        self.dist_total += distance
        
        # Update max speed
        self.max_speed = max(self.max_speed, speed)
        
        # Update average speed (simple moving average)
        if len(self.position_history) >= 2:
            total_time = self.time_history[-1] - self.time_history[0]
            if total_time > 0:
                self.avg_speed = self.dist_total / total_time
        
        # Categorize activity and update zone distances
        self._update_activity_zones(speed, distance)
        
        # Track sprints
        self._update_sprint_tracking(t_sec, speed, current_pos)
    
    def _update_activity_zones(self, speed: float, distance: float):
        """Update distance covered in different activity zones"""
        if speed <= self.walk_threshold:
            self.zone_distances['walking'] += distance
        elif speed <= self.run_threshold:
            self.zone_distances['jogging'] += distance
        elif speed <= self.sprint_threshold:
            self.zone_distances['running'] += distance
        else:
            self.zone_distances['sprinting'] += distance
    
    def _update_sprint_tracking(self, t_sec: float, speed: float, current_pos: Tuple[float, float]):
        """Track sprint periods"""
        if speed >= self.sprint_threshold:
            if not self._sprint_active:
                # Start new sprint
                self._sprint_active = True
                self._sprint_start = t_sec
                self._sprint_start_pos = current_pos
        else:
            if self._sprint_active:
                # End current sprint
                sprint_duration = t_sec - self._sprint_start
                if sprint_duration >= 0.5:  # Minimum sprint duration
                    self.sprint_windows.append((self._sprint_start, t_sec))
                    
                    # Calculate sprint distance
                    if self._sprint_start_pos:
                        dx = current_pos[0] - self._sprint_start_pos[0]
                        dy = current_pos[1] - self._sprint_start_pos[1]
                        sprint_dist = np.sqrt(dx**2 + dy**2)
                        self.sprint_distance += sprint_dist
                
                self._sprint_active = False
                self._sprint_start = None
                self._sprint_start_pos = None
    
    def get_current_speed(self) -> float:
        """Get current smoothed speed"""
        if len(self.position_history) < 2:
            return 0.0
        
        # Calculate speed from last two points
        current_pos = self.position_history[-1]
        prev_pos = self.position_history[-2]
        dt = self.time_history[-1] - self.time_history[-2]
        
        if dt <= 0:
            return 0.0
        
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        speed = np.sqrt(dx**2 + dy**2) / dt
        
        return speed
    
    def get_smoothed_speed(self) -> float:
        """Get smoothed speed over the window"""
        if len(self.position_history) < 2:
            return 0.0
        
        # Calculate average speed over the smoothing window
        total_distance = 0.0
        total_time = self.time_history[-1] - self.time_history[0]
        
        if total_time <= 0:
            return 0.0
        
        for i in range(1, len(self.position_history)):
            prev_pos = self.position_history[i-1]
            curr_pos = self.position_history[i]
            
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        return total_distance / total_time
    
    def summary(self) -> Dict:
        """Get summary of all metrics"""
        return {
            "player_id": self.player_id,
            "distance_m": self.dist_total,
            "distance_km": self.dist_total / 1000.0,
            "max_speed_mps": self.max_speed,
            "max_speed_kmh": self.max_speed * 3.6,
            "avg_speed_mps": self.avg_speed,
            "avg_speed_kmh": self.avg_speed * 3.6,
            "current_speed_mps": self.get_current_speed(),
            "current_speed_kmh": self.get_current_speed() * 3.6,
            "smoothed_speed_mps": self.get_smoothed_speed(),
            "smoothed_speed_kmh": self.get_smoothed_speed() * 3.6,
            "sprint_bursts": len(self.sprint_windows),
            "sprint_distance_m": self.sprint_distance,
            "sprint_distance_km": self.sprint_distance / 1000.0,
            "activity_zones": self.zone_distances.copy(),
            "sprint_windows": self.sprint_windows.copy()
        }

class PhysicalMetricsManager:
    """Manage physical metrics for all players"""
    
    def __init__(self):
        self.trackers: Dict[int, SpeedTracker] = {}
        self.frame_rate = 30.0  # Default FPS
        
    def update_player(self, player_id: int, t_sec: float, x_m: float, y_m: float):
        """Update metrics for a specific player"""
        if player_id not in self.trackers:
            self.trackers[player_id] = SpeedTracker(player_id)
        
        self.trackers[player_id].update(t_sec, x_m, y_m)
    
    def get_player_metrics(self, player_id: int) -> Optional[Dict]:
        """Get metrics for a specific player"""
        if player_id in self.trackers:
            return self.trackers[player_id].summary()
        return None
    
    def get_all_metrics(self) -> Dict[int, Dict]:
        """Get metrics for all players"""
        return {pid: tracker.summary() for pid, tracker in self.trackers.items()}
    
    def get_team_metrics(self, team_assignments: Dict[int, str]) -> Dict[str, Dict]:
        """Get aggregated metrics by team"""
        team_metrics = {}
        
        for player_id, tracker in self.trackers.items():
            team = team_assignments.get(player_id, "unknown")
            
            if team not in team_metrics:
                team_metrics[team] = {
                    "total_distance_m": 0.0,
                    "total_distance_km": 0.0,
                    "max_speed_mps": 0.0,
                    "max_speed_kmh": 0.0,
                    "total_sprints": 0,
                    "total_sprint_distance_m": 0.0,
                    "player_count": 0,
                    "players": []
                }
            
            summary = tracker.summary()
            team_metrics[team]["total_distance_m"] += summary["distance_m"]
            team_metrics[team]["total_distance_km"] += summary["distance_km"]
            team_metrics[team]["max_speed_mps"] = max(
                team_metrics[team]["max_speed_mps"], 
                summary["max_speed_mps"]
            )
            team_metrics[team]["max_speed_kmh"] = max(
                team_metrics[team]["max_speed_kmh"], 
                summary["max_speed_kmh"]
            )
            team_metrics[team]["total_sprints"] += summary["sprint_bursts"]
            team_metrics[team]["total_sprint_distance_m"] += summary["sprint_distance_m"]
            team_metrics[team]["player_count"] += 1
            team_metrics[team]["players"].append(summary)
        
        return team_metrics
    
    def reset_player(self, player_id: int):
        """Reset metrics for a specific player"""
        if player_id in self.trackers:
            del self.trackers[player_id]
    
    def reset_all(self):
        """Reset all metrics"""
        self.trackers.clear()

