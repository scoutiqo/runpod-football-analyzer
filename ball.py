# ball.py
import numpy as np

class BallTracker:
    def __init__(self):
        self.last = None

    def update(self, detections):
        balls = [d for d in detections if d["cls"] in (32, 37)]  # 32=coco 'sports ball'
        if not balls:
            return self.last
        # choose highest conf / closest to last
        if self.last is None:
            b = max(balls, key=lambda d: d["conf"])
        else:
            def center(d): return ((d["x1"]+d["x2"])/2, (d["y1"]+d["y2"])/2)
            bx,by = center(self.last)
            b = min(balls, key=lambda d: (center(d)[0]-bx)**2 + (center(d)[1]-by)**2 - 0.1*d["conf"])
        self.last = b
        return self.last
