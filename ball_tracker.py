# ball_tracker.py
from typing import Any, Dict, List, Optional

SPORTS_BALL_ID = 32  # COCO class id

class BallTracker:
    """
    Pick the best 'sports ball' detection each frame.
    Accepts either:
      - dict-of-arrays: {"xyxy": Nx4, "conf": N, "cls": N}
      - list-of-dicts : [{"x1","y1","x2","y2","conf","cls"}, ...]
    Returns dict {x1,y1,x2,y2,cls,conf} or None.
    """
    def __init__(self) -> None:
        self.last: Optional[Dict[str, float]] = None

    def _to_list_of_dicts(self, dets: Any) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        if dets is None:
            return out

        # Case A: dict of arrays
        if isinstance(dets, dict) and "xyxy" in dets and "cls" in dets:
            xyxy = dets.get("xyxy") or []
            conf = dets.get("conf") or []
            cls  = dets.get("cls")  or []
            n = min(len(xyxy), len(conf), len(cls))
            for i in range(n):
                c = cls[i]
                try:
                    c = int(getattr(c, "item", lambda: c)())
                except Exception:
                    continue
                if c == SPORTS_BALL_ID:
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                    out.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cls": SPORTS_BALL_ID,
                        "conf": float(conf[i]) if i < len(conf) else 0.0
                    })
            return out

        # Case B: list of dicts
        if isinstance(dets, (list, tuple)):
            for d in dets:
                if not isinstance(d, dict):
                    continue
                c = d.get("cls", d.get("class_id"))
                try:
                    c = int(getattr(c, "item", lambda: c)())
                except Exception:
                    continue
                if c == SPORTS_BALL_ID:
                    out.append({
                        "x1": float(d.get("x1", 0)), "y1": float(d.get("y1", 0)),
                        "x2": float(d.get("x2", 0)), "y2": float(d.get("y2", 0)),
                        "cls": SPORTS_BALL_ID,
                        "conf": float(d.get("conf", 0.0)),
                    })
        return out

    def update(self, detections: Any) -> Optional[Dict[str, float]]:
        balls = self._to_list_of_dicts(detections)
        if not balls:
            return None
        # pick highest confidence
        best = max(balls, key=lambda b: b.get("conf", 0.0))
        self.last = best
        return best
