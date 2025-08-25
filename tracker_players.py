# tracker_players.py
import numpy as np
import supervision as sv

class PlayerTracker:
    def __init__(self, conf_thresh: float = 0.20, person_ids=(0,)):
        # ByteTrack inside Supervision
        self.trk = sv.ByteTrack()
        self.conf_thresh = float(conf_thresh)
        self.person_ids = set(int(i) for i in person_ids)

    def _normalize(self, dets: dict):
        """
        Ensure we always return proper numpy arrays (even when empty).
        dets should have keys: xyxy, conf, cls
        """
        if dets is None:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        xyxy = dets.get("xyxy")
        conf = dets.get("conf")
        cls  = dets.get("cls")

        # Coerce None -> empty arrays
        if xyxy is None: xyxy = []
        if conf is None: conf = []
        if cls  is None: cls  = []

        xyxy = np.asarray(xyxy, dtype=np.float32)
        conf = np.asarray(conf, dtype=np.float32)
        cls  = np.asarray(cls,  dtype=np.int32)

        # If shapes are wrong, fall back to empty
        if xyxy.ndim != 2 or (xyxy.size and xyxy.shape[1] != 4):
            xyxy = np.empty((0, 4), dtype=np.float32)
            conf = np.empty((0,), dtype=np.float32)
            cls  = np.empty((0,), dtype=np.int32)

        # Lengths must match
        n = len(xyxy)
        if len(conf) != n or len(cls) != n:
            xyxy = np.empty((0, 4), dtype=np.float32)
            conf = np.empty((0,), dtype=np.float32)
            cls  = np.empty((0,), dtype=np.int32)

        return xyxy, conf, cls

    def update(self, dets: dict):
        xyxy, conf, cls = self._normalize(dets)

        # Filter to persons + confidence
        if xyxy.size:
            mask = np.isin(cls, list(self.person_ids)) & (conf >= self.conf_thresh)
            xyxy = xyxy[mask]
            conf = conf[mask]
            cls  = cls[mask]
        # Build Supervision detections (empty-safe)
        detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)

        # Track
        tracks = self.trk.update_with_detections(detections)

        out = []
        # tracks is a Detections object; it may be empty
        for i in range(len(tracks)):
            x1, y1, x2, y2 = tracks.xyxy[i].tolist()
            tid = int(tracks.tracker_id[i]) if tracks.tracker_id is not None else -1
            c   = float(tracks.confidence[i]) if hasattr(tracks, "confidence") and tracks.confidence is not None else 0.0
            kls = int(tracks.class_id[i]) if hasattr(tracks, "class_id") and tracks.class_id is not None else 0
            out.append({"id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": kls, "conf": c})
        return out
