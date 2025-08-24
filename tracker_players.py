import supervision as sv

class PlayerTracker:
    """
    ByteTrack wrapper via supervision. Tracks only persons (cls=0).
    """
    def __init__(self, track_thresh=0.5, match_thresh=0.8):
        self.tracker = sv.ByteTrack(
            track_thresh=track_thresh,
            track_buffer=30,
            match_thresh=match_thresh
        )

    def update(self, detections):
        # filter to persons
        xyxy, conf, cls = [], [], []
        for d in detections:
            if d["cls"] == 0:
                xyxy.append([d["x1"], d["y1"], d["x2"], d["y2"]])
                conf.append(d["conf"])
                cls.append(d["cls"])

        if not xyxy:
            self.tracker.update_with_detections(sv.Detections.empty())
            return []

        dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        tracks = self.tracker.update_with_detections(dets)

        out = []
        for i in range(len(tracks)):
            out.append({
                "id": int(tracks.tracker_id[i]),
                "x1": float(tracks.xyxy[i][0]), "y1": float(tracks.xyxy[i][1]),
                "x2": float(tracks.xyxy[i][2]), "y2": float(tracks.xyxy[i][3]),
                "cls": int(tracks.class_id[i]), "conf": float(tracks.confidence[i])
            })
        return out
