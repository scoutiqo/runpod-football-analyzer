# analyzers/value_models.py
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from schemas.contracts import Event, Value


class xTGrid:
    """Expected Threat (xT) grid model"""
    
    def __init__(self, grid_size: Tuple[int, int] = (16, 12)):
        self.grid_size = grid_size
        self.xT_values = self._initialize_xT_grid()
    
    def _initialize_xT_grid(self) -> np.ndarray:
        """Initialize xT grid with basic values"""
        grid = np.zeros(self.grid_size)
        
        # Higher xT values closer to goal
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Distance from goal (normalized)
                distance_from_goal = (self.grid_size[0] - i) / self.grid_size[0]
                
                # Higher xT in central areas
                center_bias = 1.0 - abs(j - self.grid_size[1] / 2) / (self.grid_size[1] / 2)
                
                # Combine factors
                grid[i, j] = distance_from_goal * center_bias * 0.1
        
        return grid
    
    def get_xT(self, x: float, y: float) -> float:
        """Get xT value for a position"""
        # Convert normalized coordinates to grid indices
        grid_x = int(x * (self.grid_size[1] - 1))
        grid_y = int(y * (self.grid_size[0] - 1))
        
        # Clamp to grid bounds
        grid_x = max(0, min(self.grid_size[1] - 1, grid_x))
        grid_y = max(0, min(self.grid_size[0] - 1, grid_y))
        
        return float(self.xT_values[grid_y, grid_x])
    
    def get_xT_delta(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate xT delta between two positions"""
        xT_start = self.get_xT(x1, y1)
        xT_end = self.get_xT(x2, y2)
        return xT_end - xT_start


class EPVModel:
    """Expected Possession Value (EPV) model"""
    
    def __init__(self):
        self.goal_value = 1.0
        self.possession_value = 0.1
    
    def get_EPV(self, x: float, y: float, has_possession: bool = True) -> float:
        """Get EPV for a position"""
        if not has_possession:
            return 0.0
        
        # Distance to goal
        distance_to_goal = np.sqrt((x - 1.0)**2 + (y - 0.5)**2)
        
        # Higher EPV closer to goal
        epv = self.goal_value * np.exp(-distance_to_goal * 2.0)
        
        return float(epv)


class VAEPModel:
    """Valuing Actions by Estimating Probabilities (VAEP) model"""
    
    def __init__(self):
        self.action_weights = {
            'pass': 0.3,
            'carry': 0.2,
            'shot': 0.8,
            'tackle': 0.4,
            'interception': 0.5
        }
    
    def get_VAEP(self, event_type: str, x: float, y: float, outcome: str) -> float:
        """Get VAEP for an event"""
        base_weight = self.action_weights.get(event_type, 0.1)
        
        # Outcome multiplier
        outcome_multiplier = 1.0
        if outcome in ['complete', 'won', 'goal']:
            outcome_multiplier = 1.2
        elif outcome in ['incomplete', 'lost']:
            outcome_multiplier = 0.8
        
        # Position multiplier (higher value in attacking areas)
        position_multiplier = 1.0 + (x * 0.5)  # Higher value closer to goal
        
        return float(base_weight * outcome_multiplier * position_multiplier)


class PackingModel:
    """Packing model - opponents bypassed"""
    
    def __init__(self):
        self.opponent_positions = []
    
    def update_opponents(self, opponents: List[Dict[str, Any]]):
        """Update opponent positions"""
        self.opponent_positions = []
        for opp in opponents:
            self.opponent_positions.append({
                'x': opp.get('x', 0.0),
                'y': opp.get('y', 0.0),
                'tid': opp.get('tid', -1)
            })
    
    def count_packing(self, x1: float, y1: float, x2: float, y2: float) -> int:
        """Count opponents bypassed between two positions"""
        if not self.opponent_positions:
            return 0
        
        bypassed = 0
        
        for opp in self.opponent_positions:
            opp_x, opp_y = opp['x'], opp['y']
            
            # Check if opponent is between start and end positions
            if self._is_between_positions(x1, y1, x2, y2, opp_x, opp_y):
                bypassed += 1
        
        return bypassed
    
    def _is_between_positions(self, x1: float, y1: float, x2: float, y2: float, 
                             opp_x: float, opp_y: float) -> bool:
        """Check if opponent is between two positions"""
        # Simple line intersection check
        # This is a simplified version - in practice, you'd use proper line geometry
        
        # Check if opponent is in the rectangle formed by the two positions
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        return min_x <= opp_x <= max_x and min_y <= opp_y <= max_y


class ValueModels:
    """Main value models class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.xT_grid = xTGrid()
        self.epv_model = EPVModel()
        self.vaep_model = VAEPModel()
        self.packing_model = PackingModel()
    
    def compute_values(self, events: List[Event], opponents: Optional[List[Dict[str, Any]]] = None) -> List[Event]:
        """
        Compute value models for events
        
        Args:
            events: List of events
            opponents: Optional list of opponent positions for packing
            
        Returns:
            List of events with enriched value data
        """
        if opponents:
            self.packing_model.update_opponents(opponents)
        
        enriched_events = []
        
        for event in events:
            # Compute xT
            xT = self.xT_grid.get_xT(event.loc.x, event.loc.y)
            
            # Compute EPV
            epv = self.epv_model.get_EPV(event.loc.x, event.loc.y, True)
            
            # Compute VAEP
            vaep = self.vaep_model.get_VAEP(event.type.value, event.loc.x, event.loc.y, event.outcome.value)
            
            # Compute packing (if we have position data)
            packing = 0
            if hasattr(event, 'to_tid') and event.to_tid:
                # This would need the actual position data
                # For now, use a simple heuristic
                packing = 1 if event.type.value == 'pass' else 0
            
            # Create enriched value
            enriched_value = Value(
                xT=xT,
                xA=0.0,  # Would need more complex model
                EPV=epv,
                VAEP=vaep
            )
            
            # Create enriched event
            enriched_event = Event(
                id=event.id,
                t=event.t,
                phase=event.phase,
                type=event.type,
                team=event.team,
                actor_tid=event.actor_tid,
                to_tid=event.to_tid,
                loc=event.loc,
                orient_deg=event.orient_deg,
                pressure=event.pressure,
                value=enriched_value,
                outcome=event.outcome,
                nextN=event.nextN
            )
            
            enriched_events.append(enriched_event)
        
        return enriched_events


def compute_values(events: List[Event], control_model: Optional[Dict[str, Any]] = None) -> List[Event]:
    """
    Main function to compute value models for events
    
    Args:
        events: List of events
        control_model: Optional control model configuration
        
    Returns:
        List of events with enriched value data
    """
    config = control_model or {}
    value_models = ValueModels(config)
    return value_models.compute_values(events)



