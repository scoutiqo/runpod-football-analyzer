# events/actions.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class Event:
    t: float
    team: str
    type: str
    actor_tid: int | None
    to_tid: int | None
    loc: Tuple[float,float]          # meters
    pressure_n: int
    nearest_def_m: float
    outcome: str | None
    tags: Dict[str, float]

def pressure_metrics(actor_tid: int, team_of: Dict[int,str], player_xy: Dict[int,Tuple[float,float]], radius=3.0):
    if actor_tid is None: return 0, 99.0
    actor_xy = np.array(player_xy.get(actor_tid))
    pres = [np.linalg.norm(np.array(xy)-actor_xy) for tid, xy in player_xy.items() if team_of.get(tid) not in (None, team_of.get(actor_tid))]
    if not pres: return 0, 99.0
    return int(np.sum(np.array(pres) <= radius)), float(np.min(pres))

def detect_actions(frames, fps: float, poss, goal_x=105.0):
    """
    Minimal CV heuristics:
    - A 'pass' when ball leaves owner and next owner within 1.5s is teammate at a different location.
    - A 'carry' between consecutive touches by same owner if ball moved > 2m.
    - A 'shot' when ball speed > v_th and trajectory heads towards goal mouth (x near 105, y in [20,48]).
    """
    events: List[Event] = []
    # index frames by f for quick lookup
    indexed = {s.f: s for s in frames}

    # build quick ball speed
    ball_xy = np.array([s.ball_xy for s in frames])
    diff = np.linalg.norm(np.diff(ball_xy, axis=0), axis=1) * fps  # m/s approximate (per frame distance * fps)
    speed = np.concatenate([[0.0], diff])

    # map frame->possession team (None if OOP)
    in_poss = {}
    for p in poss:
        for f in range(p["start_f"], p["end_f"]+1):
            in_poss[f] = p["team"]

    # detect receptions & passes
    last_owner = None; last_owner_f = None; last_owner_xy = None
    for s in frames:
        owner_tid = None
        # owner = nearest in 1.8m if in_possession
        team = in_poss.get(s.f)
        if team is not None:
            # nearest player of that team
            cands = [(tid,xy) for tid,xy in s.player_xy.items() if s.team_of.get(tid)==team]
            if cands:
                tids = np.array([t for t,_ in cands]); pts = np.array([p for _,p in cands])
                d = np.linalg.norm(pts - np.array(s.ball_xy), axis=1)
                i = int(np.argmin(d))
                if d[i] <= 1.8: owner_tid = int(tids[i])

        # owner changed → pass reception
        if owner_tid is not None and last_owner is not None and owner_tid != last_owner:
            # classify previous as pass if time delta < 1.5s and distance > 3m
            dt = (s.f - last_owner_f) / fps
            dist = float(np.linalg.norm(np.array(s.ball_xy) - np.array(last_owner_xy)))
            if dt <= 1.5 and dist >= 3.0:
                pn, nd = pressure_metrics(last_owner, s.team_of, s.player_xy)
                events.append(Event(
                    t=s.f/fps, team=team, type="pass", actor_tid=last_owner, to_tid=owner_tid,
                    loc=last_owner_xy, pressure_n=pn, nearest_def_m=nd,
                    outcome="complete", tags={"dist_m": dist}
                ))
        # shot detection
        bx,by = s.ball_xy
        if speed[s.f] >= 18.0 and bx > (goal_x - 20.0) and 20.0 <= by <= 48.0:
            pn, nd = pressure_metrics(owner_tid, s.team_of, s.player_xy)
            events.append(Event(
                t=s.f/fps, team=team, type="shot", actor_tid=owner_tid, to_tid=None,
                loc=s.ball_xy, pressure_n=pn, nearest_def_m=nd, outcome=None, tags={"v_ms": speed[s.f]}
            ))

        # carries (owner keeps ball and ball moved)
        if owner_tid is not None and last_owner == owner_tid and last_owner_xy is not None:
            dist = float(np.linalg.norm(np.array(s.ball_xy)-np.array(last_owner_xy)))
            if dist >= 1.5:
                pn, nd = pressure_metrics(owner_tid, s.team_of, s.player_xy)
                events.append(Event(
                    t=s.f/fps, team=team, type="carry", actor_tid=owner_tid, to_tid=None,
                    loc=last_owner_xy, pressure_n=pn, nearest_def_m=nd, outcome="progress", tags={"dist_m": dist}
                ))

        if owner_tid is not None:
            last_owner, last_owner_f, last_owner_xy = owner_tid, s.f, s.ball_xy

    return events
