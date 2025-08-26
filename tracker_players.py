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
            cfg = conf_thresh
            conf_thresh = cfg.get("conf_thresh", 0.20)
            person_ids  = cfg.get("person_ids", person_ids)

        self.trk = sv.ByteTrack()  # defaults are fine for now
        self.conf_thresh = float(conf_thresh)
        self.person_ids = set(int(i) for i in person_ids)

    def _normalize(self, dets: dict):
        """Return numpy arrays (xyxy Nx4, conf N, cls N). Empty-safe."""
        if dets is None:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        xyxy = dets.get("xyxy") or []
        conf = dets.get("conf") or []
        cls  = dets.get("cls")  or []

        xyxy = np.asarray(xyxy, dtype=np.float32)
        conf = np.asarray(conf, dtype=np.float32)
        cls  = np.asarray(cls,  dtype=np.int32)

        # shape checks
        if xyxy.ndim != 2 or (xyxy.size and xyxy.shape[1] != 4):
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        n = len(xyxy)
        if len(conf) != n or len(cls) != n:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        return xyxy, conf, cls

    def update(self, dets: dict):
        xyxy, conf, cls = self._normalize(dets)

        # keep only persons and above confidence
        if xyxy.size:
            mask = np.isin(cls, list(self.person_ids)) & (conf >= self.conf_thresh)
            xyxy = xyxy[mask]; conf = conf[mask]; cls = cls[mask]

        detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        tracks = self.trk.update_with_detections(detections)

        out = []
        for i in range(len(tracks)):
            x1, y1, x2, y2 = tracks.xyxy[i].tolist()
            tid = int(tracks.tracker_id[i]) if tracks.tracker_id is not None else -1
            c   = float(tracks.confidence[i]) if getattr(tracks, "confidence", None) is not None else 0.0
            kls = int(tracks.class_id[i]) if getattr(tracks, "class_id", None) is not None else 0
            out.append({"id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": kls, "conf": c})
        return out
