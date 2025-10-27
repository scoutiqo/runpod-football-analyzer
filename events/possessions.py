# events/possessions.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

@dataclass
class FrameState:
    f: int
    ball_xy: Tuple[float,float]          # meters
    player_xy: Dict[int, Tuple[float,float]]  # tid -> (x,y)
    team_of: Dict[int, str]              # tid -> "home"/"away"

def nearest_player(ball_xy, player_xy) -> int | None:
    if not player_xy: return None
    tids = np.array(list(player_xy.keys()))
    pts = np.array(list(player_xy.values()))
    d = np.linalg.norm(pts - np.array(ball_xy), axis=1)
    i = int(np.argmin(d))
    return int(tids[i])

def build_possessions(frames: List[FrameState], max_gap=15, control_radius=2.5):
    """
    Returns: list of possessions [{team, start_f, end_f, touches: [(f, tid), ...]}]
    Heuristic control: ball nearest player within control_radius meters.
    """
    poss = []
    cur = None
    last_touch_f = None

    for s in frames:
        owner_tid = nearest_player(s.ball_xy, s.player_xy)
        owner_team = s.team_of.get(owner_tid) if owner_tid is not None else None
        # in control if the owner tid is within radius
        in_ctrl = False
        if owner_tid is not None:
            d = np.linalg.norm(np.array(s.player_xy[owner_tid]) - np.array(s.ball_xy))
            in_ctrl = d <= control_radius

        if cur is None:
            if owner_team and in_ctrl:
                cur = {"team": owner_team, "start_f": s.f, "end_f": s.f, "touches": [(s.f, owner_tid)]}
                last_touch_f = s.f
            continue

        # same team keeps control unless long gap without touch
        if owner_team == cur["team"] and in_ctrl:
            cur["end_f"] = s.f
            if last_touch_f is None or (s.f - last_touch_f) > 3:
                cur["touches"].append((s.f, owner_tid))
                last_touch_f = s.f
        else:
            # turnover or long gap → close possession if lasted > minimal frames
            if cur["end_f"] - cur["start_f"] >= 5:
                poss.append(cur)
            cur = None
            last_touch_f = None

    if cur is not None and cur["end_f"] - cur["start_f"] >= 5:
        poss.append(cur)
    return poss
