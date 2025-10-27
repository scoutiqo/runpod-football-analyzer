# analyzers/ball_tracker.py
from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple

class BallTracker:
    """
    Tracks a single ball as a point (cx, cy).
    Primary: YOLO detections (COCO 'sports ball' id=32).
    Fallback: Lucas–Kanade optical flow from last known location.
    """
    def __init__(self, max_flow_gap: int = 20):
        self.point: Optional[np.ndarray] = None   # shape (1,1,2) float32
        self.prev_gray: Optional[np.ndarray] = None
        self.missed = 0
        self.max_flow_gap = max_flow_gap

    @staticmethod
    def _to_point(bbox_xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        return np.array([[[ (x1+x2)/2.0, (y1+y2)/2.0 ]]], dtype=np.float32)

    def update_with_detection(self, frame_bgr: np.ndarray, bbox_xyxy: np.ndarray):
        self.point = self._to_point(bbox_xyxy)
        self.prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.missed = 0

    def update_optical_flow(self, frame_bgr: np.ndarray) -> Optional[Tuple[float,float]]:
        if self.point is None or self.prev_gray is None:
            # nothing to track yet
            self.prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            return None
        if self.missed > self.max_flow_gap:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        nextPt, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.point, None,
                                                   winSize=(21,21), maxLevel=3,
                                                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.prev_gray = gray
        if st is not None and st[0][0] == 1:
            self.point = nextPt
            self.missed += 1
            return float(self.point[0,0,0]), float(self.point[0,0,1])
        return None

    def current_xy(self) -> Optional[Tuple[float,float]]:
        if self.point is None:
            return None
        return float(self.point[0,0,0]), float(self.point[0,0,1])
