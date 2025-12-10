# analyzers/events.py
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from schemas.contracts import Event, EventType, PhaseType, TeamType, OutcomeType, Location, Pressure, Value, NextN


class EventDetector:
    """Detects football events from tracking data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pass_distance_threshold = config.get('pass_distance_threshold', 5.0)  # meters
        self.carry_distance_threshold = config.get('carry_distance_threshold', 3.0)  # meters
        self.shot_distance_threshold = config.get('shot_distance_threshold', 20.0)  # meters
        self.pressure_distance = config.get('pressure_distance', 2.0)  # meters
        self.min_event_duration = config.get('min_event_duration', 0.5)  # seconds
    
    def extract_events(self, tracks: Dict[str, Any]) -> List[Event]:
        """
        Extract events from tracking data
        
        Args:
            tracks: TrackingResult object with player and ball data
            
        Returns:
            List of detected events
        """
        events = []
        
        # Get player and ball data
        players = tracks.get('players', [])
        ball = tracks.get('ball', [])
        
        if not players or not ball:
            return events
        
        # Group data by timestamp
        player_by_time = self._group_by_time(players)
        ball_by_time = self._group_by_time(ball)
        
        # Detect different event types
        events.extend(self._detect_passes(player_by_time, ball_by_time))
        events.extend(self._detect_carries(player_by_time, ball_by_time))
        events.extend(self._detect_shots(player_by_time, ball_by_time))
        events.extend(self._detect_tackles(player_by_time, ball_by_time))
        events.extend(self._detect_interceptions(player_by_time, ball_by_time))
        
        # Sort events by timestamp
        events.sort(key=lambda x: x.t)
        
        return events
    
    def _group_by_time(self, data: List[Dict[str, Any]]) -> Dict[float, List[Dict[str, Any]]]:
        """Group tracking data by timestamp"""
        grouped = {}
        for item in data:
            timestamp = item.get('timestamp', 0.0)
            if timestamp not in grouped:
                grouped[timestamp] = []
            grouped[timestamp].append(item)
        return grouped
    
    def _detect_passes(self, player_by_time: Dict[float, List[Dict]], ball_by_time: Dict[float, List[Dict]]) -> List[Event]:
        """Detect pass events"""
        events = []
        
        # Simple heuristic: ball moves between players
        ball_positions = []
        for timestamp, ball_data in ball_by_time.items():
            if ball_data:
                ball_pos = ball_data[0]  # Take first ball detection
                ball_positions.append({
                    'timestamp': timestamp,
                    'x': (ball_pos['x1'] + ball_pos['x2']) / 2,
                    'y': (ball_pos['y1'] + ball_pos['y2']) / 2
                })
        
        # Find ball movements that could be passes
        for i in range(1, len(ball_positions)):
            prev_pos = ball_positions[i-1]
            curr_pos = ball_positions[i]
            
            # Calculate movement distance
            distance = np.sqrt((curr_pos['x'] - prev_pos['x'])**2 + (curr_pos['y'] - prev_pos['y'])**2)
            
            if distance > self.pass_distance_threshold:
                # Find closest players at start and end of movement
                start_player = self._find_closest_player(prev_pos, player_by_time.get(prev_pos['timestamp'], []))
                end_player = self._find_closest_player(curr_pos, player_by_time.get(curr_pos['timestamp'], []))
                
                if start_player and end_player and start_player['player_id'] != end_player['player_id']:
                    # Create pass event
                    event = Event(
                        id=len(events),
                        t=curr_pos['timestamp'],
                        phase=PhaseType.BUILD,  # Default phase
                        type=EventType.PASS,
                        team=TeamType.UNKNOWN,  # Would need team assignment
                        actor_tid=start_player['player_id'],
                        to_tid=end_player['player_id'],
                        loc=Location(x=curr_pos['x'] / 1920, y=curr_pos['y'] / 1080),  # Normalize
                        pressure=Pressure(count_n=0, nearest_m=0.0),
                        value=Value(),
                        outcome=OutcomeType.COMPLETE,
                        nextN=NextN()
                    )
                    events.append(event)
        
        return events
    
    def _detect_carries(self, player_by_time: Dict[float, List[Dict]], ball_by_time: Dict[float, List[Dict]]) -> List[Event]:
        """Detect carry events"""
        events = []
        
        # Track ball possession by players
        possession_changes = []
        
        for timestamp in sorted(player_by_time.keys()):
            players = player_by_time[timestamp]
            ball_data = ball_by_time.get(timestamp, [])
            
            if not ball_data:
                continue
            
            ball_pos = ball_data[0]
            ball_center = ((ball_pos['x1'] + ball_pos['x2']) / 2, (ball_pos['y1'] + ball_pos['y2']) / 2)
            
            # Find player closest to ball
            closest_player = self._find_closest_player({'x': ball_center[0], 'y': ball_center[1]}, players)
            
            if closest_player:
                possession_changes.append({
                    'timestamp': timestamp,
                    'player_id': closest_player['player_id'],
                    'x': ball_center[0],
                    'y': ball_center[1]
                })
        
        # Detect carry sequences
        current_carry = None
        for change in possession_changes:
            if current_carry is None:
                current_carry = change
            elif current_carry['player_id'] == change['player_id']:
                # Continue carry
                current_carry = change
            else:
                # End previous carry, start new one
                if current_carry:
                    # Create carry event
                    event = Event(
                        id=len(events),
                        t=current_carry['timestamp'],
                        phase=PhaseType.BUILD,
                        type=EventType.CARRY,
                        team=TeamType.UNKNOWN,
                        actor_tid=current_carry['player_id'],
                        loc=Location(x=current_carry['x'] / 1920, y=current_carry['y'] / 1080),
                        pressure=Pressure(count_n=0, nearest_m=0.0),
                        value=Value(),
                        outcome=OutcomeType.COMPLETE,
                        nextN=NextN()
                    )
                    events.append(event)
                
                current_carry = change
        
        return events
    
    def _detect_shots(self, player_by_time: Dict[float, List[Dict]], ball_by_time: Dict[float, List[Dict]]) -> List[Event]:
        """Detect shot events"""
        events = []
        
        # Simple heuristic: ball moves quickly towards goal
        ball_positions = []
        for timestamp, ball_data in ball_by_time.items():
            if ball_data:
                ball_pos = ball_data[0]
                ball_positions.append({
                    'timestamp': timestamp,
                    'x': (ball_pos['x1'] + ball_pos['x2']) / 2,
                    'y': (ball_pos['y1'] + ball_pos['y2']) / 2
                })
        
        # Find rapid ball movements (potential shots)
        for i in range(1, len(ball_positions)):
            prev_pos = ball_positions[i-1]
            curr_pos = ball_positions[i]
            
            # Calculate speed
            time_diff = curr_pos['timestamp'] - prev_pos['timestamp']
            if time_diff > 0:
                distance = np.sqrt((curr_pos['x'] - prev_pos['x'])**2 + (curr_pos['y'] - prev_pos['y'])**2)
                speed = distance / time_diff
                
                # High speed movement could be a shot
                if speed > 50:  # pixels per second threshold
                    # Find closest player
                    closest_player = self._find_closest_player(curr_pos, player_by_time.get(curr_pos['timestamp'], []))
                    
                    if closest_player:
                        event = Event(
                            id=len(events),
                            t=curr_pos['timestamp'],
                            phase=PhaseType.BUILD,
                            type=EventType.SHOT,
                            team=TeamType.UNKNOWN,
                            actor_tid=closest_player['player_id'],
                            loc=Location(x=curr_pos['x'] / 1920, y=curr_pos['y'] / 1080),
                            pressure=Pressure(count_n=0, nearest_m=0.0),
                            value=Value(),
                            outcome=OutcomeType.SHOT_OFF,  # Default
                            nextN=NextN()
                        )
                        events.append(event)
        
        return events
    
    def _detect_tackles(self, player_by_time: Dict[float, List[Dict]], ball_by_time: Dict[float, List[Dict]]) -> List[Event]:
        """Detect tackle events"""
        events = []
        
        # Simple heuristic: player gets close to ball-carrying opponent
        for timestamp in sorted(player_by_time.keys()):
            players = player_by_time[timestamp]
            ball_data = ball_by_time.get(timestamp, [])
            
            if not ball_data or len(players) < 2:
                continue
            
            ball_pos = ball_data[0]
            ball_center = ((ball_pos['x1'] + ball_pos['x2']) / 2, (ball_pos['y1'] + ball_pos['y2']) / 2)
            
            # Find ball carrier
            ball_carrier = self._find_closest_player({'x': ball_center[0], 'y': ball_center[1]}, players)
            
            if ball_carrier:
                # Find other players close to ball carrier
                for player in players:
                    if player['player_id'] != ball_carrier['player_id']:
                        distance = np.sqrt(
                            (player['x1'] - ball_carrier['x1'])**2 + 
                            (player['y1'] - ball_carrier['y1'])**2
                        )
                        
                        # Close proximity could indicate tackle attempt
                        if distance < self.pressure_distance * 50:  # Convert to pixels
                            event = Event(
                                id=len(events),
                                t=timestamp,
                                phase=PhaseType.DEFEND,
                                type=EventType.TACKLE,
                                team=TeamType.UNKNOWN,
                                actor_tid=player['player_id'],
                                loc=Location(x=player['x1'] / 1920, y=player['y1'] / 1080),
                                pressure=Pressure(count_n=1, nearest_m=distance / 50),
                                value=Value(),
                                outcome=OutcomeType.WON,
                                nextN=NextN()
                            )
                            events.append(event)
        
        return events
    
    def _detect_interceptions(self, player_by_time: Dict[float, List[Dict]], ball_by_time: Dict[float, List[Dict]]) -> List[Event]:
        """Detect interception events"""
        events = []
        
        # Simple heuristic: player gains possession when not closest to ball initially
        for timestamp in sorted(player_by_time.keys()):
            players = player_by_time[timestamp]
            ball_data = ball_by_time.get(timestamp, [])
            
            if not ball_data or len(players) < 2:
                continue
            
            ball_pos = ball_data[0]
            ball_center = ((ball_pos['x1'] + ball_pos['x2']) / 2, (ball_pos['y1'] + ball_pos['y2']) / 2)
            
            # Find closest player to ball
            closest_player = self._find_closest_player({'x': ball_center[0], 'y': ball_center[1]}, players)
            
            if closest_player:
                # Check if this player was not the closest in previous frame
                prev_timestamp = timestamp - 0.1  # 100ms ago
                prev_players = player_by_time.get(prev_timestamp, [])
                
                if prev_players:
                    prev_closest = self._find_closest_player({'x': ball_center[0], 'y': ball_center[1]}, prev_players)
                    
                    if prev_closest and prev_closest['player_id'] != closest_player['player_id']:
                        # Possession changed - potential interception
                        event = Event(
                            id=len(events),
                            t=timestamp,
                            phase=PhaseType.DEFEND,
                            type=EventType.INTERCEPTION,
                            team=TeamType.UNKNOWN,
                            actor_tid=closest_player['player_id'],
                            loc=Location(x=ball_center[0] / 1920, y=ball_center[1] / 1080),
                            pressure=Pressure(count_n=0, nearest_m=0.0),
                            value=Value(),
                            outcome=OutcomeType.WON,
                            nextN=NextN()
                        )
                        events.append(event)
        
        return events
    
    def _find_closest_player(self, position: Dict[str, float], players: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the player closest to a given position"""
        if not players:
            return None
        
        min_distance = float('inf')
        closest_player = None
        
        for player in players:
            player_center_x = (player['x1'] + player['x2']) / 2
            player_center_y = (player['y1'] + player['y2']) / 2
            
            distance = np.sqrt(
                (position['x'] - player_center_x)**2 + 
                (position['y'] - player_center_y)**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_player = player
        
        return closest_player


def extract_events(tracks: Dict[str, Any], cfg: Dict[str, Any]) -> List[Event]:
    """
    Main function to extract events from tracking data
    
    Args:
        tracks: TrackingResult object
        cfg: Configuration dictionary
        
    Returns:
        List of detected events
    """
    detector = EventDetector(cfg.get('events', {}))
    return detector.extract_events(tracks)
