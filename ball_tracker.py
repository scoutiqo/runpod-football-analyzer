print(">>> BallTracker v2 loaded", flush=True)
# ball_tracker.py
from typing import Any, Dict, List, Optional
SPORTS_BALL_ID = 32  # COCO ball class

class BallTracker:
    def __init__(self, cfg: dict | None = None) -> None:
        self.last: Optional[Dict[str, float]] = None
        self.cls_id = (cfg or {}).get("class_id", SPORTS_BALL_ID)
        self.min_conf = float((cfg or {}).get("min_conf", 0.20))

    def _to_list_of_dicts(self, dets: Any) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        if dets is None:
            return out

        # Case A: dict-of-arrays (preferred)
        if isinstance(dets, dict) and "xyxy" in dets and "cls" in dets:
            xyxy = dets.get("xyxy") or []
            conf = dets.get("conf") or []
            cls  = dets.get("cls")  or []
            n = min(len(xyxy), len(conf), len(cls))
            for i in range(n):
                val = cls[i]
                c = int(getattr(val, "item", lambda: val)()) if val is not None else -1
                if c == self.cls_id and (conf[i] if i < len(conf) else 0.0) >= self.min_conf:
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                    out.append({"x1":x1, "y1":y1, "x2":x2, "y2":y2, "cls":c, "conf":float(conf[i])})
            return out

        # Case B: list-of-dicts
        if isinstance(dets, (list, tuple)):
            for d in dets:
                if not isinstance(d, dict): continue
                val = d.get("cls", d.get("class_id", d.get("class", -1)))
                c = int(getattr(val, "item", lambda: val)())
                if c == self.cls_id and float(d.get("conf", 0.0)) >= self.min_conf:
                    out.append({
                        "x1": float(d.get("x1", 0)), "y1": float(d.get("y1", 0)),
                        "x2": float(d.get("x2", 0)), "y2": float(d.get("y2", 0)),
                        "cls": c, "conf": float(d.get("conf", 0.0))
                    })
        return out

    def update(self, detections: Any) -> Optional[Dict[str, float]]:
        balls = self._to_list_of_dicts(detections)
        if not balls:
            return None
        best = max(balls, key=lambda b: b.get("conf", 0.0))
        self.last = best
        return best
