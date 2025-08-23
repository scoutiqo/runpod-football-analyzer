# detector.py
from ultralytics import YOLO

class Detector:
    def __init__(self, model_name="yolov8n.pt", conf=0.35):
        self.model = YOLO(model_name)
        self.conf = conf

    def infer(self, frame_bgr):
        """Return detections as list of dicts: {cls, conf, x1,y1,x2,y2}."""
        res = self.model.predict(frame_bgr, verbose=False, conf=self.conf)[0]
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cls = int(b.cls[0])
            x1,y1,x2,y2 = b.xyxy[0].tolist()
            out.append({"cls": cls, "conf": float(b.conf[0]),
                        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
        return out
