# tracker.py
import supervision as sv

class PlayerTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections):
        """
        detections: list[{cls, conf, x1,y1,x2,y2}]
        returns list of track dicts: {id, x1,y1,x2,y2, cls, conf}
        """
        # filter to persons only (COCO 0)
        xyxy = []
        conf = []
        cls  = []
        for d in detections:
            if d["cls"] == 0: # person
                xyxy.append([d["x1"], d["y1"], d["x2"], d["y2"]])
                conf.append(d["conf"])
                cls.append(d["cls"])
        if not xyxy:
            self.tracker.update_with_detections(sv.Detections.empty())
            return []
        dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
        tracks = self.tracker.update_with_detections(dets)
        out=[]
        for i in range(len(tracks)):
            out.append({
                "id": int(tracks.tracker_id[i]),
                "x1": float(tracks.xyxy[i][0]), "y1": float(tracks.xyxy[i][1]),
                "x2": float(tracks.xyxy[i][2]), "y2": float(tracks.xyxy[i][3]),
                "cls": int(tracks.class_id[i]), "conf": float(tracks.confidence[i])
            })
        return out
