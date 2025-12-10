# render/heatmap.py
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter


class HeatmapGenerator:
    """Generates player heatmaps from tracking data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pitch_width = config.get('pitch_width', 1920)
        self.pitch_height = config.get('pitch_height', 1080)
        self.heatmap_resolution = config.get('heatmap_resolution', (100, 100))
        self.gaussian_sigma = config.get('gaussian_sigma', 2.0)
        self.colormap = config.get('colormap', 'hot')
    
    def player_heatmap(self, tracks: Dict[str, Any], player_id: int, 
                      output_path: Optional[str] = None) -> str:
        """
        Generate heatmap for a specific player
        
        Args:
            tracks: Tracking data
            player_id: Player tracking ID
            output_path: Optional output path for the heatmap
            
        Returns:
            Path to generated heatmap file
        """
        # Extract player positions
        player_positions = self._extract_player_positions(tracks, player_id)
        
        if not player_positions:
            # Create empty heatmap
            heatmap = np.zeros(self.heatmap_resolution)
        else:
            # Generate heatmap
            heatmap = self._generate_heatmap(player_positions)
        
        # Create visualization
        if output_path is None:
            output_path = f"heatmap_player_{player_id}.png"
        
        self._save_heatmap(heatmap, output_path)
        
        return output_path
    
    def _extract_player_positions(self, tracks: Dict[str, Any], player_id: int) -> List[Tuple[float, float]]:
        """Extract player positions from tracking data"""
        positions = []
        
        players_data = tracks.get('players', [])
        for frame_data in players_data:
            if frame_data.get('player_id') == player_id:
                x = frame_data.get('x1', 0) + (frame_data.get('x2', 0) - frame_data.get('x1', 0)) / 2
                y = frame_data.get('y1', 0) + (frame_data.get('y2', 0) - frame_data.get('y1', 0)) / 2
                positions.append((x, y))
        
        return positions
    
    def _generate_heatmap(self, positions: List[Tuple[float, float]]) -> np.ndarray:
        """Generate heatmap from positions"""
        # Create grid
        heatmap = np.zeros(self.heatmap_resolution)
        
        # Convert positions to grid coordinates
        for x, y in positions:
            # Normalize coordinates
            norm_x = x / self.pitch_width
            norm_y = y / self.pitch_height
            
            # Convert to grid indices
            grid_x = int(norm_x * (self.heatmap_resolution[1] - 1))
            grid_y = int(norm_y * (self.heatmap_resolution[0] - 1))
            
            # Clamp to grid bounds
            grid_x = max(0, min(self.heatmap_resolution[1] - 1, grid_x))
            grid_y = max(0, min(self.heatmap_resolution[0] - 1, grid_y))
            
            # Add to heatmap
            heatmap[grid_y, grid_x] += 1
        
        # Apply Gaussian filter for smoothing
        heatmap = gaussian_filter(heatmap, sigma=self.gaussian_sigma)
        
        return heatmap
    
    def _save_heatmap(self, heatmap: np.ndarray, output_path: str):
        """Save heatmap as PNG file"""
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap
        im = ax.imshow(heatmap, cmap=self.colormap, origin='lower', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Activity')
        
        # Add pitch outline
        self._add_pitch_outline(ax)
        
        # Set labels
        ax.set_xlabel('Pitch Width')
        ax.set_ylabel('Pitch Height')
        ax.set_title('Player Heatmap')
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _add_pitch_outline(self, ax):
        """Add football pitch outline to the plot"""
        # Simple pitch outline
        # This would be more sophisticated in practice
        pitch_rect = patches.Rectangle((0, 0), self.heatmap_resolution[1], 
                                      self.heatmap_resolution[0], 
                                      linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(pitch_rect)
    
    def create_overlay_video(self, video_path: str, tracks: Dict[str, Any], 
                           events: List[Dict[str, Any]], output_path: str):
        """
        Create overlay video with tracking and events
        
        Args:
            video_path: Path to input video
            tracks: Tracking data
            events: List of events
            output_path: Path to output video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add tracking overlays
                frame = self._add_tracking_overlay(frame, tracks, frame_idx)
                
                # Add event overlays
                frame = self._add_event_overlay(frame, events, frame_idx)
                
                # Write frame
                out.write(frame)
                frame_idx += 1
        
        finally:
            cap.release()
            out.release()
    
    def _add_tracking_overlay(self, frame: np.ndarray, tracks: Dict[str, Any], 
                             frame_idx: int) -> np.ndarray:
        """Add tracking overlay to frame"""
        # Get player positions for this frame
        players_data = tracks.get('players', [])
        frame_players = [p for p in players_data if p.get('frame_idx') == frame_idx]
        
        # Draw player bounding boxes
        for player in frame_players:
            x1, y1 = int(player.get('x1', 0)), int(player.get('y1', 0))
            x2, y2 = int(player.get('x2', 0)), int(player.get('y2', 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw player ID
            cv2.putText(frame, str(player.get('player_id', '')), 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw ball
        ball_data = tracks.get('ball', [])
        frame_ball = [b for b in ball_data if b.get('frame_idx') == frame_idx]
        
        if frame_ball:
            ball = frame_ball[0]
            x1, y1 = int(ball.get('x1', 0)), int(ball.get('y1', 0))
            x2, y2 = int(ball.get('x2', 0)), int(ball.get('y2', 0))
            
            # Draw ball
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return frame
    
    def _add_event_overlay(self, frame: np.ndarray, events: List[Dict[str, Any]], 
                          frame_idx: int) -> np.ndarray:
        """Add event overlay to frame"""
        # Find events for this frame
        frame_events = [e for e in events if e.get('frame_idx') == frame_idx]
        
        # Draw event markers
        for event in frame_events:
            x = int(event.get('x', 0))
            y = int(event.get('y', 0))
            event_type = event.get('type', '')
            
            # Draw event marker
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            
            # Draw event type
            cv2.putText(frame, event_type, (x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return frame


def player_heatmap(tracks: Dict[str, Any], player_id: int, 
                  output_path: Optional[str] = None) -> str:
    """
    Main function to generate player heatmap
    
    Args:
        tracks: Tracking data
        player_id: Player tracking ID
        output_path: Optional output path
        
    Returns:
        Path to generated heatmap file
    """
    config = {}
    generator = HeatmapGenerator(config)
    return generator.player_heatmap(tracks, player_id, output_path)



