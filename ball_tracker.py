class BallTracker:
    """
    Simple ball tracker: pick 'sports ball' det (cls=32); prefer proximity to last + confidence.
    """
    def __init__(self):
        self.last = None

    @staticmethod
    def center(d): return ((d["x1"]+d["x2"])/2.0, (d["y1"]+d["y2"])/2.0)

    def update(self, detections):
        balls = [d for d in detections if d["cls"] == 32]
        if not balls:
            return self.last  # keep last seen
        if self.last is None:
            b = max(balls, key=lambda z: z["conf"])
        else:
            lx,ly = self.center(self.last)
            def score(d):
                x,y = self.center(d)
                # closer to last, higher conf -> better
                return (x-lx)**2 + (y-ly)**2 - 2000.0*d["conf"]
            b = min(balls, key=score)
        self.last = b
        return self.last
