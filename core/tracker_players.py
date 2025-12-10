import numpy as np
import supervision as sv
import json
import os
from pathlib import Path

CONFIG_FILE = "models/tracker_config.json"

class PlayerTracker:
    """
    ByteTrack wrapper using Supervision.
    Robust against empty frames and array shape mismatches.
    """
    def __init__(self, conf_thresh=0.20, person_ids=(0,), **kwargs):
        # 1. Handle direct config dict
        if isinstance(conf_thresh, dict):
            cfg = conf_thresh
            conf_thresh = cfg.get("conf_thresh", 0.20)
        
        # 2. Load AI-Tuned Config (if available)
        # This overrides the default, making the tracker "Smart"
        if os.path.exists(CONFIG_FILE):
            try:
                cfg = json.loads(Path(CONFIG_FILE).read_text())
                conf_thresh = cfg.get("conf_thresh", conf_thresh)
                # We can also tune buffer size later
            except: pass

        self.conf_thresh = float(conf_thresh)
        self.person_ids = set(int(i) for i in person_ids)

        # STICKY SETTINGS: Keep tracks alive for 60 frames (approx 2 sec)
        self.trk = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,
            minimum_matching_threshold=0.8
        )
    
    # ... (Rest of the file stays the same: _normalize and update methods) ...

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

        if xyxy is None: xyxy = np.empty((0, 4), dtype=np.float32)
        else: xyxy = np.asarray(xyxy, dtype=np.float32)

        if conf is None: conf = np.empty((0,), dtype=np.float32)
        else: conf = np.asarray(conf, dtype=np.float32)

        if cls is None: cls = np.empty((0,), dtype=np.int32)
        else: cls = np.asarray(cls, dtype=np.int32)

        # Guards
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
        if xyxy.size > 0:
            mask = np.isin(cls, list(self.person_ids)) & (conf >= self.conf_thresh)
            xyxy = xyxy[mask]
            conf = conf[mask]
            cls = cls[mask]

        # Create Detections object
        # Handle empty case explicitly to avoid ByteTrack internal errors
        if len(xyxy) == 0:
            try:
                # Pass empty to keep tracker state updating (expiration)
                self.trk.update_with_detections(sv.Detections.empty())
            except: pass
            return []

        detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        
        try:
            tracks = self.trk.update_with_detections(detections)
        except Exception:
            return []

        # Robust extraction from Supervision Detections
        t_xyxy = getattr(tracks, "xyxy", None)
        t_ids  = getattr(tracks, "tracker_id", None)
        
        if t_xyxy is None or t_ids is None:
            return []

        # Convert to numpy and flatten
        t_xyxy = np.asarray(t_xyxy)
        t_ids  = np.asarray(t_ids).flatten()
        
        # Safety Checks
        if t_xyxy.ndim != 2 or t_xyxy.shape[1] != 4:
            return []
        if t_ids.ndim != 1:
            return []

        n = min(len(t_xyxy), len(t_ids))
        if n == 0:
            return []

        out = []
        for i in range(n):
            try:
                # Explicit index check just in case
                if i >= len(t_ids): break
                
                x1, y1, x2, y2 = t_xyxy[i].tolist()
                tid = int(t_ids[i])
                
                out.append({
                    "id": tid,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cls": 0,
                    "conf": 1.0 # Tracked objects are assumed valid
                })
            except Exception:
                continue
                
        return out
