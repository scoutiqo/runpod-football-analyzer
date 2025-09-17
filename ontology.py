# ontology.py (Your Code-First Schema)
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class BodyPart(str, Enum):
    LEFT_FOOT = "left_foot"
    RIGHT_FOOT = "right_foot"
    HEAD = "head"
    OTHER = "other"

class PassHeight(str, Enum):
    GROUND = "ground"
    LOW = "low"
    HIGH = "high"

class SetPiece(str, Enum):
    NONE = "none"
    CORNER = "corner"
    FK_DIRECT = "fk_direct"
    FK_INDIRECT = "fk_indirect"
    THROW_IN = "throw_in"
    PENALTY = "penalty"
    KICK_OFF = "kick_off"

class EventType(str, Enum):
    PASS = "pass"
    CARRY = "carry"
    DRIBBLE = "dribble"
    SHOT = "shot"
    PRESSURE = "pressure"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    BLOCK_SHOT = "block_shot"
    BLOCK_PASS = "block_pass"
    CLEARANCE = "clearance"
    AERIAL_DUEL = "aerial_duel"
    FOUL = "foul"
    OFFSIDE = "offside"
    RECOVERY = "recovery"
    TURNOVER = "turnover"
    SET_PIECE = "set_piece"
    GK_SAVE = "gk_save"
    GK_CLAIM = "gk_claim"
    GK_PUNCH = "gk_punch"
    GK_DISTRIBUTION = "gk_distribution"

class XY(BaseModel):
    x: float  # meters from own goal line
    y: float  # meters from left touchline

class BaseEvent(BaseModel):
    event_id: str
    type: EventType
    t_start: float
    t_end: float
    period: int
    team_id: int
    player_id: Optional[int] = None
    sequence_id: Optional[str] = None
    start_xy: Optional[XY] = None
    end_xy: Optional[XY] = None
    set_piece: SetPiece = SetPiece.NONE
    under_pressure: Optional[bool] = None
    vendor_tags: Dict[str, str] = Field(default_factory=dict)  # e.g., {"statsbomb": "Pressure", "wyscout": "Pressing"}

class PassEvent(BaseEvent):
    receiver_id: Optional[int] = None
    length_m: Optional[float] = None
    angle_rad: Optional[float] = None
    height: Optional[PassHeight] = None
    body_part: Optional[BodyPart] = None
    is_cross: bool = False
    is_cutback: bool = False
    is_through: bool = False
    progressive: bool = False
    line_breaks: int = 0
    pass_lane_risk: Optional[float] = None  # 0..1 (interception risk)
    xa: Optional[float] = None  # expected assist (if leads to shot)

class CarryEvent(BaseEvent):
    distance_m: Optional[float] = None
    progressive: bool = False
    beats_opponent: bool = False
    touches: Optional[int] = None

class ShotEvent(BaseEvent):
    body_part: Optional[BodyPart] = None
    shot_distance_m: Optional[float] = None
    shot_angle_rad: Optional[float] = None
    xg: Optional[float] = None
    xgot: Optional[float] = None
    psxg: Optional[float] = None
    blocked: bool = False
    on_target: bool = False
    goal: bool = False
    assisted_by: Optional[int] = None
    pre_assist_by: Optional[int] = None

class PressureEvent(BaseEvent):
    target_player_id: Optional[int] = None
    radius_m: Optional[float] = None
    success_turnover: bool = False

class DefensiveEvent(BaseEvent):
    won: Optional[bool] = None
    target_player_id: Optional[int] = None

class Possession(BaseModel):
    sequence_id: str
    team_id: int
    start_t: float
    end_t: float
    start_reason: str  # recovery/interception/restart
    end_reason: str    # shot/turnover/out/foul
    xg_chain: float
    actions: List[str]  # event_ids

class FrameState(BaseModel):
    t: float
    ball_xy: XY
    player_xy: Dict[int, XY]  # id -> XY
    player_speed: Dict[int, float]
    has_possession: Optional[int] = None

class ValueModelOutputs(BaseModel):
    xT_before: float
    xT_after: float
    EPV_before: float
    EPV_after: float
    OBV_delta: float
    VAEP_delta: float
