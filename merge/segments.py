# merge/segments.py
import json
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path
from schemas.contracts import TracksJSON, PlayerInsights, Player, Event, Video, Calibration, Artifacts, AutoTune
from metrics.players import aggregate_player_metrics
from render.heatmap import player_heatmap


class SegmentMerger:
    """Merges per-segment outputs into job-level tracks.json"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.job_id = str(uuid.uuid4())
    
    def merge_all(self, job_id: str, segment_results: List[Dict[str, Any]], 
                  video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge all segment results into final tracks.json
        
        Args:
            job_id: Job ID
            segment_results: List of segment results
            video_metadata: Video metadata
            
        Returns:
            Merged tracks.json data
        """
        # Initialize merged data
        merged_players = {}
        merged_events = []
        merged_artifacts = Artifacts()
        auto_tune_data = AutoTune()
        
        # Process each segment
        for i, segment_result in enumerate(segment_results):
            self._merge_segment(segment_result, merged_players, merged_events, 
                              merged_artifacts, auto_tune_data, i)
        
        # Convert players dict to list
        players_list = []
        for tid, player_data in merged_players.items():
            players_list.append(player_data)
        
        # Create video metadata
        video = Video(
            duration_s=video_metadata.get('duration_s', 0.0),
            width=video_metadata.get('width', 1920),
            height=video_metadata.get('height', 1080),
            fps=video_metadata.get('fps', 30.0)
        )
        
        # Create calibration
        calibration = Calibration(
            homography=None,  # Would be merged from segments
            units="px"
        )
        
        # Create tracks.json
        tracks_json = TracksJSON(
            job_id=job_id,
            video=video,
            calibration=calibration,
            players=players_list,
            events=merged_events,
            auto_tune=auto_tune_data,
            artifacts=merged_artifacts
        )
        
        return tracks_json.dict()
    
    def _merge_segment(self, segment_result: Dict[str, Any], merged_players: Dict[int, Player],
                      merged_events: List[Event], merged_artifacts: Artifacts,
                      auto_tune_data: AutoTune, segment_idx: int):
        """Merge a single segment result"""
        
        # Merge players
        segment_players = segment_result.get('players', [])
        for player_data in segment_players:
            tid = player_data.get('tid')
            if tid not in merged_players:
                # Create new player
                merged_players[tid] = Player(
                    tid=tid,
                    team=player_data.get('team', 'unknown'),
                    jersey=player_data.get('jersey'),
                    role_hint=player_data.get('role_hint'),
                    primary_position=player_data.get('primary_position'),
                    metrics=player_data.get('metrics', {}),
                    heatmap="",  # Will be generated later
                    events_idx=[]
                )
            
            # Merge metrics
            self._merge_player_metrics(merged_players[tid], player_data)
        
        # Merge events
        segment_events = segment_result.get('events', [])
        for event_data in segment_events:
            # Adjust event ID to be unique across segments
            event_data['id'] = len(merged_events)
            merged_events.append(Event(**event_data))
        
        # Merge artifacts
        segment_artifacts = segment_result.get('artifacts', {})
        if 'overlays' in segment_artifacts:
            merged_artifacts.overlays.extend(segment_artifacts['overlays'])
        if 'logs' in segment_artifacts:
            merged_artifacts.logs.extend(segment_artifacts['logs'])
        
        # Merge auto-tune data
        segment_auto_tune = segment_result.get('auto_tune', {})
        if segment_auto_tune:
            auto_tune_data.params.update(segment_auto_tune.get('params', {}))
            auto_tune_data.notes += f"Segment {segment_idx}: {segment_auto_tune.get('notes', '')}\n"
    
    def _merge_player_metrics(self, merged_player: Player, segment_player: Dict[str, Any]):
        """Merge player metrics from segment"""
        segment_metrics = segment_player.get('metrics', {})
        merged_metrics = merged_player.metrics
        
        # Add segment metrics to merged metrics
        merged_metrics.touches_total += segment_metrics.get('touches_total', 0)
        merged_metrics.passes.attempts += segment_metrics.get('passes', {}).get('attempts', 0)
        merged_metrics.passes.completed += segment_metrics.get('passes', {}).get('completed', 0)
        merged_metrics.passes.progressive += segment_metrics.get('passes', {}).get('progressive', 0)
        merged_metrics.carries.count += segment_metrics.get('carries', {}).get('count', 0)
        merged_metrics.carries.prog_dist_px += segment_metrics.get('carries', {}).get('prog_dist_px', 0)
        merged_metrics.dribbles.att += segment_metrics.get('dribbles', {}).get('att', 0)
        merged_metrics.dribbles.won += segment_metrics.get('dribbles', {}).get('won', 0)
        merged_metrics.shots.att += segment_metrics.get('shots', {}).get('att', 0)
        merged_metrics.shots.on_target += segment_metrics.get('shots', {}).get('on_target', 0)
        merged_metrics.shots.goals += segment_metrics.get('shots', {}).get('goals', 0)
        merged_metrics.shots.xg_sum += segment_metrics.get('shots', {}).get('xg_sum', 0)
        merged_metrics.pressures.total += segment_metrics.get('pressures', {}).get('total', 0)
        merged_metrics.pressures.successful += segment_metrics.get('pressures', {}).get('successful', 0)
        merged_metrics.duels.air_won += segment_metrics.get('duels', {}).get('air_won', 0)
        merged_metrics.duels.air_att += segment_metrics.get('duels', {}).get('air_att', 0)
        merged_metrics.duels.grd_won += segment_metrics.get('duels', {}).get('grd_won', 0)
        merged_metrics.duels.grd_att += segment_metrics.get('duels', {}).get('grd_att', 0)
        merged_metrics.distance_px += segment_metrics.get('distance_px', 0)
        merged_metrics.sprints += segment_metrics.get('sprints', 0)
        merged_metrics.max_speed_pxps = max(merged_metrics.max_speed_pxps, 
                                          segment_metrics.get('max_speed_pxps', 0))
    
    def generate_player_insights(self, job_id: str, tracks_data: Dict[str, Any], 
                                events: List[Event]) -> Dict[int, PlayerInsights]:
        """
        Generate player insights for all players
        
        Args:
            job_id: Job ID
            tracks_data: Merged tracks data
            events: List of events
            
        Returns:
            Dictionary mapping player ID to insights
        """
        player_insights = {}
        
        # Get player metrics
        player_metrics = aggregate_player_metrics(tracks_data, events)
        
        # Generate insights for each player
        for tid, metrics in player_metrics.items():
            # Generate heatmap
            heatmap_path = player_heatmap(tracks_data, tid)
            
            # Create player insights
            insights = PlayerInsights(
                job_id=job_id,
                tid=tid,
                identity={},
                minutes={},
                formations=[],
                opponent_matchups=[],
                game_states=[],
                summary={
                    'touches': metrics.touches_total,
                    'passes_att': metrics.passes.attempts,
                    'passes_cmp': metrics.passes.completed,
                    'xA': 0.0,
                    'carries': metrics.carries.count,
                    'prog_carry_px': metrics.carries.prog_dist_px,
                    'dribbles_won': metrics.dribbles.won,
                    'shots': metrics.shots.att,
                    'goals': metrics.shots.goals,
                    'xG': metrics.shots.xg_sum,
                    'psxg_on_target': 0.0,
                    'pressures': metrics.pressures.total,
                    'press_success': metrics.pressures.successful,
                    'duels_air_won_pct': self._calculate_percentage(metrics.duels.air_won, metrics.duels.air_att),
                    'duels_grd_won_pct': self._calculate_percentage(metrics.duels.grd_won, metrics.duels.grd_att),
                    'distance_px': metrics.distance_px,
                    'sprints': metrics.sprints,
                    'max_speed_pxps': metrics.max_speed_pxps
                },
                value_models={
                    'xT_sum': 0.0,
                    'EPV_sum': 0.0,
                    'VAEP_sum': 0.0,
                    'packing_for': 0,
                    'packing_against': 0
                },
                role_specific_notes=[]
            )
            
            player_insights[tid] = insights
        
        return player_insights
    
    def _calculate_percentage(self, numerator: int, denominator: int) -> float:
        """Calculate percentage safely"""
        if denominator == 0:
            return 0.0
        return (numerator / denominator) * 100.0
    
    def save_tracks_json(self, job_id: str, tracks_data: Dict[str, Any], 
                        output_dir: str) -> str:
        """
        Save tracks.json to file
        
        Args:
            job_id: Job ID
            tracks_data: Tracks data
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir) / f"tracks.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(tracks_data, f, indent=2)
        
        return str(output_path)
    
    def save_player_insights(self, player_insights: Dict[int, PlayerInsights], 
                           output_dir: str) -> List[str]:
        """
        Save player insights to files
        
        Args:
            player_insights: Player insights data
            output_dir: Output directory
            
        Returns:
            List of saved file paths
        """
        output_paths = []
        players_dir = Path(output_dir) / "players"
        players_dir.mkdir(parents=True, exist_ok=True)
        
        for tid, insights in player_insights.items():
            output_path = players_dir / f"tid_{tid}.json"
            
            with open(output_path, 'w') as f:
                json.dump(insights.dict(), f, indent=2)
            
            output_paths.append(str(output_path))
        
        return output_paths


def merge_all(job_id: str, segment_results: List[Dict[str, Any]], 
              video_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to merge all segments
    
    Args:
        job_id: Job ID
        segment_results: List of segment results
        video_metadata: Video metadata
        
    Returns:
        Merged tracks.json data
    """
    config = {}
    merger = SegmentMerger(config)
    return merger.merge_all(job_id, segment_results, video_metadata)



