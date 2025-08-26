print(">>> BallTracker v2 loaded", flush=True)

from typing import Any, Dict, List, Optional

SPORTS_BALL_ID = 32  # COCO “sports ball”

class BallTracker:
    def __init__(self, cfg: Optional[dict] = None) -> None:
        cfg = cfg or {}
        self.last: Optional[Dict[str, float]] = None
        self.cls_id = int(cfg.get("class_id", SPORTS_BALL_ID))
        self.min_conf = float(cfg.get("min_conf", 0.20))

    def _to_list_of_dicts(self, dets: Any) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        if dets is None:
            return out

        # Case A: dict-of-arrays {"xyxy": Nx4, "conf": N, "cls": N}
        if isinstance(dets, dict) and "xyxy" in dets and "cls" in dets:
            xyxy = dets.get("xyxy") or []
            conf = dets.get("conf") or []
            cls  = dets.get("cls")  or []
            n = min(len(xyxy), len(conf), len(cls))
            for i in range(n):
                val = cls[i]
                c = int(getattr(val, "item", lambda: val)()) if val is not None else -1
                score = float(conf[i]) if i < len(conf) else 0.0
                if c == self.cls_id and score >= self.min_conf:
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                    out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": c, "conf": score})
            return out

        # Case B: list-of-dicts [{"x1","y1","x2","y2","conf","cls"}, ...]
        if isinstance(dets, (list, tuple)):
            for d in dets:
                if not isinstance(d, dict):
                    continue
                val = d.get("cls", d.get("class_id", d.get("class", -1)))
                try:
                    c = int(getattr(val, "item", lambda: val)()) if val is not None else -1
                except Exception:
                    c = -1
                score = float(d.get("conf", 0.0))
                if c == self.cls_id and score >= self.min_conf:
                    out.append({
                        "x1": float(d.get("x1", 0.0)),
                        "y1": float(d.get("y1", 0.0)),
                        "x2": float(d.get("x2", 0.0)),
                        "y2": float(d.get("y2", 0.0)),
                        "cls": c,
                        "conf": score,
                    })
        return out

    def update(self, detections: Any) -> Optional[Dict[str, float]]:
        balls = self._to_list_of_dicts(detections)
        if not balls:
            return None
        best = max(balls, key=lambda b: b.get("conf", 0.0))
        self.last = best
        return best
