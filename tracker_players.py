# tracker_players.py
import numpy as np
import supervision as sv

class PlayerTracker:
    """
    ByteTrack wrapper using Supervision.
    Accepts either numeric args or a config dict, e.g.:
      PlayerTracker(0.25)  or  PlayerTracker({"conf_thresh":0.25, "person_ids":[0]})
    """
    def __init__(self,
                 conf_thresh: float | dict = 0.20,
                 person_ids=(0,),
                 **kwargs):
        # Allow the first arg to be a config dict
        if isinstance(conf_thresh, dict):
            cfg = conf_thresh or {}
            conf_thresh = cfg.get("conf_thresh", 0.20)
            person_ids  = cfg.get("person_ids", person_ids)

        self.trk = sv.ByteTrack()  # defaults are fine
        self.conf_thresh = float(conf_thresh)
        self.person_ids = set(int(i) for i in person_ids)

    def _normalize(self, dets: dict):
        """Return numpy arrays (xyxy Nx4, conf N, cls N). Empty-safe."""
        if not dets or not isinstance(dets, dict):
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        xyxy = dets.get("xyxy", None)
        conf = dets.get("conf", None)
        cls  = dets.get("cls",  None)

        # Avoid ambiguous truth checks on numpy arrays
        if xyxy is None:
            xyxy = np.empty((0, 4), dtype=np.float32)
        else:
            xyxy = np.asarray(xyxy, dtype=np.float32)

        if conf is None:
            conf = np.empty((0,), dtype=np.float32)
        else:
            conf = np.asarray(conf, dtype=np.float32)

        if cls is None:
            cls = np.empty((0,), dtype=np.int32)
        else:
            cls = np.asarray(cls, dtype=np.int32)

        # shape/length guards
        ok = xyxy.ndim == 2 and (xyxy.size == 0 or xyxy.shape[1] == 4)
        ok = ok and (len(conf) == len(xyxy)) and (len(cls) == len(xyxy))
        if not ok:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        return xyxy, conf, cls

    def update(self, dets: dict):
        xyxy, conf, cls = self._normalize(dets)

        # Filter to persons + confidence
        if xyxy.size:
            mask = np.isin(cls, list(self.person_ids)) & (conf >= self.conf_thresh)
            xyxy = xyxy[mask]; conf = conf[mask]; cls = cls[mask]

        detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        tracks = self.trk.update_with_detections(detections)

        # Robust extraction from Supervision Detections
        t_xyxy = getattr(tracks, "xyxy", None)
        t_ids  = getattr(tracks, "tracker_id", None)
        if t_xyxy is None or t_ids is None:
            return []

        t_xyxy = np.asarray(t_xyxy)
        t_ids  = np.asarray(t_ids)

        n = min(len(t_xyxy), len(t_ids))
        if n == 0:
            return []

        # Optional arrays
        t_conf = getattr(tracks, "confidence", None)
        t_cls  = getattr(tracks, "class_id",  None)
        if t_conf is not None: t_conf = np.asarray(t_conf)
        if t_cls  is not None: t_cls  = np.asarray(t_cls)

        out = []
        for i in range(n):
            # Guard per-index too
            if i >= len(t_xyxy) or i >= len(t_ids):
                continue
            x1, y1, x2, y2 = t_xyxy[i].tolist()
            tid = int(t_ids[i]) if t_ids[i] is not None else -1
            c   = float(t_conf[i]) if t_conf is not None and i < len(t_conf) else 0.0
            kls = int(t_cls[i])  if t_cls  is not None and i < len(t_cls)  else 0
            out.append({"id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": kls, "conf": c})
        return out
