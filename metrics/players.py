# metrics/players.py
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from schemas.contracts import PlayerMetrics, Player, Event, EventType, TeamType


class PlayerMetricsAggregator:
    """Aggregates player metrics from tracking data and events"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pitch_width = config.get('pitch_width', 1920)  # pixels
        self.pitch_height = config.get('pitch_height', 1080)  # pixels
        self.fps = config.get('fps', 30.0)
    
    def aggregate_player_metrics(self, tracks: Dict[str, Any], events: List[Event]) -> Dict[int, PlayerMetrics]:
        """
        Aggregate player metrics from tracks and events
        
        Args:
            tracks: Tracking data with player positions
            events: List of events
            
        Returns:
            Dictionary mapping player ID to metrics
        """
        player_metrics = defaultdict(lambda: PlayerMetrics())
        
        # Process tracking data
        self._process_tracking_data(tracks, player_metrics)
        
        # Process events
        self._process_events(events, player_metrics)
        
        return dict(player_metrics)
    
    def _process_tracking_data(self, tracks: Dict[str, Any], player_metrics: Dict[int, PlayerMetrics]):
        """Process tracking data to extract basic metrics"""
        players_data = tracks.get('players', [])
        
        if not players_data:
            return
        
        # Group by player ID
        player_positions = defaultdict(list)
        for frame_data in players_data:
            player_id = frame_data.get('player_id')
            if player_id is not None:
                player_positions[player_id].append(frame_data)
        
        # Calculate metrics for each player
        for player_id, positions in player_positions.items():
            if not positions:
                continue
            
            # Sort by timestamp
            positions.sort(key=lambda x: x.get('timestamp', 0))
            
            # Calculate distance traveled
            total_distance = 0.0
            max_speed = 0.0
            sprint_count = 0
            
            for i in range(1, len(positions)):
                prev_pos = positions[i-1]
                curr_pos = positions[i]
                
                # Calculate distance
                dx = curr_pos.get('x1', 0) - prev_pos.get('x1', 0)
                dy = curr_pos.get('y1', 0) - prev_pos.get('y1', 0)
                distance = np.sqrt(dx**2 + dy**2)
                total_distance += distance
                
                # Calculate speed
                time_diff = curr_pos.get('timestamp', 0) - prev_pos.get('timestamp', 0)
                if time_diff > 0:
                    speed = distance / time_diff
                    max_speed = max(max_speed, speed)
                    
                    # Count sprints (speed > threshold)
                    if speed > 50:  # pixels per second threshold
                        sprint_count += 1
            
            # Update metrics
            player_metrics[player_id].distance_px = total_distance
            player_metrics[player_id].max_speed_pxps = max_speed
            player_metrics[player_id].sprints = sprint_count
            player_metrics[player_id].touches_total = len(positions)
    
    def _process_events(self, events: List[Event], player_metrics: Dict[int, PlayerMetrics]):
        """Process events to extract event-based metrics"""
        for event in events:
            player_id = event.actor_tid
            
            if event.type == EventType.PASS:
                player_metrics[player_id].passes.attempts += 1
                if event.outcome.value == 'complete':
                    player_metrics[player_id].passes.completed += 1
                
                # Check if pass is progressive (moves ball forward)
                if self._is_progressive_pass(event):
                    player_metrics[player_id].passes.progressive += 1
            
            elif event.type == EventType.CARRY:
                player_metrics[player_id].carries.count += 1
                # Calculate progressive carry distance
                if hasattr(event, 'to_tid') and event.to_tid:
                    # This would need actual position data
                    player_metrics[player_id].carries.prog_dist_px += 10.0  # Placeholder
            
            elif event.type == EventType.DRIBBLE:
                player_metrics[player_id].dribbles.att += 1
                if event.outcome.value == 'won':
                    player_metrics[player_id].dribbles.won += 1
            
            elif event.type == EventType.SHOT:
                player_metrics[player_id].shots.att += 1
                if event.outcome.value in ['shot_on', 'goal']:
                    player_metrics[player_id].shots.on_target += 1
                if event.outcome.value == 'goal':
                    player_metrics[player_id].shots.goals += 1
                
                # Add xG (simplified)
                xG = self._calculate_xG(event.loc.x, event.loc.y)
                player_metrics[player_id].shots.xg_sum += xG
            
            elif event.type == EventType.TACKLE:
                player_metrics[player_id].duels.grd_att += 1
                if event.outcome.value == 'won':
                    player_metrics[player_id].duels.grd_won += 1
            
            elif event.type == EventType.INTERCEPTION:
                player_metrics[player_id].duels.air_att += 1
                if event.outcome.value == 'won':
                    player_metrics[player_id].duels.air_won += 1
            
            elif event.type == EventType.PRESS:
                player_metrics[player_id].pressures.total += 1
                if event.outcome.value == 'won':
                    player_metrics[player_id].pressures.successful += 1
    
    def _is_progressive_pass(self, event: Event) -> bool:
        """Check if a pass is progressive (moves ball forward)"""
        # Simplified heuristic: passes towards goal are progressive
        # In practice, this would need actual position data
        return event.loc.x > 0.5  # Passes in attacking half
    
    def _calculate_xG(self, x: float, y: float) -> float:
        """Calculate expected goals (xG) for a shot"""
        # Simplified xG model based on position
        # Distance from goal
        distance = np.sqrt((x - 1.0)**2 + (y - 0.5)**2)
        
        # Angle to goal
        angle = np.arctan2(abs(y - 0.5), 1.0 - x)
        
        # Basic xG calculation
        xG = 0.5 * np.exp(-distance * 2.0) * np.cos(angle)
        
        return max(0.0, min(1.0, xG))


class PlayerInsightsGenerator:
    """Generates comprehensive player insights"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def generate_player_insights(self, job_id: str, player_id: int, 
                                metrics: PlayerMetrics, events: List[Event],
                                tracks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive player insights
        
        Args:
            job_id: Job ID
            player_id: Player tracking ID
            metrics: Player metrics
            events: List of events
            tracks: Tracking data
            
        Returns:
            Player insights dictionary
        """
        # Filter events for this player
        player_events = [e for e in events if e.actor_tid == player_id]
        
        # Calculate summary statistics
        summary = self._calculate_summary(metrics, player_events)
        
        # Calculate value models
        value_models = self._calculate_value_models(player_events)
        
        # Generate role-specific notes
        role_notes = self._generate_role_notes(metrics, player_events)
        
        return {
            'job_id': job_id,
            'tid': player_id,
            'identity': {
                'name': None,
                'squad_number': None,
                'foot': None
            },
            'minutes': {
                'on': 0.0,
                'off': None,
                'played': 0.0
            },
            'formations': [],
            'opponent_matchups': [],
            'game_states': [],
            'summary': summary,
            'value_models': value_models,
            'role_specific_notes': role_notes
        }
    
    def _calculate_summary(self, metrics: PlayerMetrics, events: List[Event]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        return {
            'touches': metrics.touches_total,
            'passes_att': metrics.passes.attempts,
            'passes_cmp': metrics.passes.completed,
            'xA': 0.0,  # Would need more complex calculation
            'carries': metrics.carries.count,
            'prog_carry_px': metrics.carries.prog_dist_px,
            'dribbles_won': metrics.dribbles.won,
            'shots': metrics.shots.att,
            'goals': metrics.shots.goals,
            'xG': metrics.shots.xg_sum,
            'psxg_on_target': 0.0,  # Would need more complex calculation
            'pressures': metrics.pressures.total,
            'press_success': metrics.pressures.successful,
            'duels_air_won_pct': self._calculate_percentage(metrics.duels.air_won, metrics.duels.air_att),
            'duels_grd_won_pct': self._calculate_percentage(metrics.duels.grd_won, metrics.duels.grd_att),
            'distance_px': metrics.distance_px,
            'sprints': metrics.sprints,
            'max_speed_pxps': metrics.max_speed_pxps
        }
    
    def _calculate_value_models(self, events: List[Event]) -> Dict[str, Any]:
        """Calculate value model statistics"""
        xT_sum = sum(e.value.xT for e in events)
        EPV_sum = sum(e.value.EPV for e in events)
        VAEP_sum = sum(e.value.VAEP for e in events)
        
        return {
            'xT_sum': xT_sum,
            'EPV_sum': EPV_sum,
            'VAEP_sum': VAEP_sum,
            'packing_for': 0,  # Would need more complex calculation
            'packing_against': 0
        }
    
    def _calculate_percentage(self, numerator: int, denominator: int) -> float:
        """Calculate percentage safely"""
        if denominator == 0:
            return 0.0
        return (numerator / denominator) * 100.0
    
    def _generate_role_notes(self, metrics: PlayerMetrics, events: List[Event]) -> List[str]:
        """Generate role-specific notes"""
        notes = []
        
        # Passing notes
        if metrics.passes.attempts > 0:
            completion_rate = metrics.passes.completed / metrics.passes.attempts
            if completion_rate > 0.8:
                notes.append(f"High pass completion rate: {completion_rate:.1%}")
        
        # Shooting notes
        if metrics.shots.att > 0:
            shot_accuracy = metrics.shots.on_target / metrics.shots.att
            if shot_accuracy > 0.5:
                notes.append(f"Good shot accuracy: {shot_accuracy:.1%}")
        
        # Defensive notes
        if metrics.duels.grd_att > 0:
            tackle_success = metrics.duels.grd_won / metrics.duels.grd_att
            if tackle_success > 0.6:
                notes.append(f"Strong tackling: {tackle_success:.1%} success rate")
        
        return notes


def aggregate_player_metrics(tracks: Dict[str, Any], events: List[Event]) -> Dict[int, PlayerMetrics]:
    """
    Main function to aggregate player metrics
    
    Args:
        tracks: Tracking data
        events: List of events
        
    Returns:
        Dictionary mapping player ID to metrics
    """
    config = {}
    aggregator = PlayerMetricsAggregator(config)
    return aggregator.aggregate_player_metrics(tracks, events)



