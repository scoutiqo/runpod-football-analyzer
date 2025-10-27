# metrics/enrich.py
from __future__ import annotations
from typing import List, Dict, Tuple
from events.actions import Event
from metrics.packing import count_packing
from metrics.value_models import value_pass

def enrich_events(events: List[Event], frames_by_f, team_of, attack_dir: Dict[str,int]):
    """
    attack_dir: team -> +1 if attacking towards x=105, -1 if towards x=0 (we flip coords for packing).
    """
    out = []
    for ev in events:
        d = ev.__dict__.copy()
        if ev.type == "pass" and ev.actor_tid is not None and ev.to_tid is not None:
            # gather teammates/opponents x,y at pass start frame
            s = frames_by_f[int(ev.t * frames_by_f["fps"])]  # approximate
            team = ev.team
            opp_xy = {tid:xy for tid,xy in s.player_xy.items() if team_of.get(tid) not in (None, team)}
            # packing (flip if team attacks left)
            pack = count_packing(ev.loc, s.player_xy[ev.to_tid], opp_xy, goal_x=105.0 if attack_dir.get(team,1)>0 else 0.0)
            val = value_pass(ev.loc, s.player_xy[ev.to_tid], ev.pressure_n, ev.nearest_def_m)
            d["tags"].update({"packing": pack, **val})
        out.append(d)
    return out
