# schemas/contracts.py
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field
from enum import Enum


class TeamType(str, Enum):
    HOME = "home"
    AWAY = "away"
    UNKNOWN = "unknown"


class PhaseType(str, Enum):
    BUILD = "build"
    DEFEND = "defend"
    PRESS = "press"
    TRANSITION = "transition"


class EventType(str, Enum):
    PASS = "pass"
    CARRY = "carry"
    SHOT = "shot"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    RECEPTION = "reception"
    DRIBBLE = "dribble"
    CROSS = "cross"
    FOUL = "foul"
    RECOVERY = "recovery"
    CLEARANCE = "clearance"
    PRESS = "press"
    COUNTERPRESS = "counterpress"
    SETPIECE = "setpiece"
    GOAL = "goal"
    SAVE = "save"


class OutcomeType(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    WON = "won"
    LOST = "lost"
    SHOT_ON = "shot_on"
    SHOT_OFF = "shot_off"
    GOAL = "goal"
    BLOCKED = "blocked"
    FOUL_WON = "foul_won"
    FOUL_COMMITTED = "foul_committed"


class Location(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized x coordinate (0-1)")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized y coordinate (0-1)")


class Pressure(BaseModel):
    count_n: int = Field(..., ge=0, description="Number of pressers within N meters")
    nearest_m: float = Field(..., ge=0.0, description="Distance to nearest defender in meters")


class Value(BaseModel):
    xT: float = Field(0.0, description="Expected Threat value")
    xA: float = Field(0.0, description="Expected Assist value")
    EPV: float = Field(0.0, description="Expected Possession Value")
    VAEP: float = Field(0.0, description="Valuing Actions by Estimating Probabilities")


class NextN(BaseModel):
    shot: bool = Field(False, description="Whether next event is a shot")
    regain_s: float = Field(0.0, description="Time to regain possession in seconds")
    turnover: bool = Field(False, description="Whether possession was turned over")


class Passes(BaseModel):
    attempts: int = Field(0, ge=0)
    completed: int = Field(0, ge=0)
    progressive: int = Field(0, ge=0)


class Carries(BaseModel):
    count: int = Field(0, ge=0)
    prog_dist_px: float = Field(0.0, ge=0.0)


class Dribbles(BaseModel):
    att: int = Field(0, ge=0)
    won: int = Field(0, ge=0)


class Shots(BaseModel):
    att: int = Field(0, ge=0)
    on_target: int = Field(0, ge=0)
    goals: int = Field(0, ge=0)
    xg_sum: float = Field(0.0, ge=0.0)


class Pressures(BaseModel):
    total: int = Field(0, ge=0)
    successful: int = Field(0, ge=0)


class Duels(BaseModel):
    air_won: int = Field(0, ge=0)
    air_att: int = Field(0, ge=0)
    grd_won: int = Field(0, ge=0)
    grd_att: int = Field(0, ge=0)


class PlayerMetrics(BaseModel):
    touches_total: int = Field(0, ge=0)
    passes: Passes = Field(default_factory=Passes)
    carries: Carries = Field(default_factory=Carries)
    dribbles: Dribbles = Field(default_factory=Dribbles)
    shots: Shots = Field(default_factory=Shots)
    pressures: Pressures = Field(default_factory=Pressures)
    duels: Duels = Field(default_factory=Duels)
    distance_px: float = Field(0.0, ge=0.0)
    sprints: int = Field(0, ge=0)
    max_speed_pxps: float = Field(0.0, ge=0.0)


class Player(BaseModel):
    tid: int = Field(..., description="Player tracking ID")
    team: TeamType = Field(..., description="Team assignment")
    jersey: Optional[int] = Field(None, description="Jersey number if detected")
    role_hint: Optional[str] = Field(None, description="Positional role hint")
    primary_position: Optional[str] = Field(None, description="Primary position")
    metrics: PlayerMetrics = Field(default_factory=PlayerMetrics)
    heatmap: str = Field(..., description="Signed URL to heatmap PNG")
    events_idx: List[int] = Field(default_factory=list, description="Indices of events involving this player")


class Video(BaseModel):
    duration_s: float = Field(..., ge=0.0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    fps: float = Field(..., gt=0.0)


class Calibration(BaseModel):
    homography: Optional[List[List[float]]] = Field(None, description="3x3 homography matrix")
    units: str = Field("px", description="Distance units (px or m)")


class Event(BaseModel):
    id: int = Field(..., description="Event ID")
    t: float = Field(..., ge=0.0, description="Timestamp in seconds")
    phase: PhaseType = Field(..., description="Game phase")
    type: EventType = Field(..., description="Event type")
    team: TeamType = Field(..., description="Team in possession")
    actor_tid: int = Field(..., description="Actor player ID")
    to_tid: Optional[int] = Field(None, description="Target player ID for passes")
    loc: Location = Field(..., description="Event location")
    orient_deg: Optional[float] = Field(None, description="Player orientation in degrees")
    pressure: Pressure = Field(default_factory=Pressure)
    value: Value = Field(default_factory=Value)
    outcome: OutcomeType = Field(..., description="Event outcome")
    nextN: NextN = Field(default_factory=NextN)


class AutoTune(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)
    notes: str = Field("", description="AI loop decisions")


class Artifacts(BaseModel):
    overlays: List[str] = Field(default_factory=list, description="Signed URLs to overlay videos")
    logs: List[str] = Field(default_factory=list, description="Signed URLs to log files")


class TracksJSON(BaseModel):
    job_id: str = Field(..., description="Job UUID")
    video: Video = Field(..., description="Video metadata")
    calibration: Calibration = Field(..., description="Calibration data")
    players: List[Player] = Field(..., description="Player data with metrics")
    events: List[Event] = Field(..., description="Event data")
    auto_tune: AutoTune = Field(default_factory=AutoTune)
    artifacts: Artifacts = Field(default_factory=Artifacts)


# Player Insights JSON Schema
class Identity(BaseModel):
    name: Optional[str] = Field(None)
    squad_number: Optional[int] = Field(None)
    foot: Optional[str] = Field(None)


class Minutes(BaseModel):
    on: float = Field(0.0, ge=0.0, description="Minute came on")
    off: Optional[float] = Field(None, description="Minute came off")
    played: float = Field(0.0, ge=0.0, description="Minutes played")


class Formation(BaseModel):
    ts: float = Field(..., ge=0.0, description="Timestamp")
    shape: str = Field(..., description="Formation shape")
    role: str = Field(..., description="Player role in formation")


class OpponentMatchup(BaseModel):
    phase: PhaseType = Field(..., description="Game phase")
    vs_tid: int = Field(..., description="Opponent player ID")


class GameState(BaseModel):
    minute: int = Field(..., ge=0, description="Game minute")
    state: str = Field(..., description="Score state (level/leading/trailing)")


class Summary(BaseModel):
    touches: int = Field(0, ge=0)
    passes_att: int = Field(0, ge=0)
    passes_cmp: int = Field(0, ge=0)
    xA: float = Field(0.0, ge=0.0)
    carries: int = Field(0, ge=0)
    prog_carry_px: float = Field(0.0, ge=0.0)
    dribbles_won: int = Field(0, ge=0)
    shots: int = Field(0, ge=0)
    goals: int = Field(0, ge=0)
    xG: float = Field(0.0, ge=0.0)
    psxg_on_target: float = Field(0.0, ge=0.0)
    pressures: int = Field(0, ge=0)
    press_success: int = Field(0, ge=0)
    duels_air_won_pct: float = Field(0.0, ge=0.0, le=100.0)
    duels_grd_won_pct: float = Field(0.0, ge=0.0, le=100.0)
    distance_px: float = Field(0.0, ge=0.0)
    sprints: int = Field(0, ge=0)
    max_speed_pxps: float = Field(0.0, ge=0.0)


class ValueModels(BaseModel):
    xT_sum: float = Field(0.0, ge=0.0)
    EPV_sum: float = Field(0.0, ge=0.0)
    VAEP_sum: float = Field(0.0, ge=0.0)
    packing_for: int = Field(0, ge=0)
    packing_against: int = Field(0, ge=0)


class PlayerInsights(BaseModel):
    job_id: str = Field(..., description="Job UUID")
    tid: int = Field(..., description="Player tracking ID")
    identity: Identity = Field(default_factory=Identity)
    minutes: Minutes = Field(default_factory=Minutes)
    formations: List[Formation] = Field(default_factory=list)
    opponent_matchups: List[OpponentMatchup] = Field(default_factory=list)
    game_states: List[GameState] = Field(default_factory=list)
    summary: Summary = Field(default_factory=Summary)
    value_models: ValueModels = Field(default_factory=ValueModels)
    role_specific_notes: List[str] = Field(default_factory=list)


# Tracking Result Schema
class TrackingResult(BaseModel):
    players: List[Dict[str, Any]] = Field(..., description="Player tracking data")
    ball: List[Dict[str, Any]] = Field(..., description="Ball tracking data")
    frames: List[Dict[str, Any]] = Field(..., description="Frame metadata")
    homography: Optional[List[List[float]]] = Field(None, description="Calibration matrix")



