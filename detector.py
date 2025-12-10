# detector.py
import os, tempfile, requests
from ultralytics import YOLO
from config import sign_storage_url

class Detector:
    def __init__(self, cfg: dict):
        dcfg = cfg.get("detector", {})
        self.conf = float(dcfg.get("conf", 0.25))
        self.iou  = float(dcfg.get("iou", 0.45))
        self.weights_url = dcfg.get("weights_url")            # optional direct URL
        self.bucket = dcfg.get("weights_bucket", "models")
        self.path   = dcfg.get("weights_path", "prod/best.pt")
        import os
self.model = YOLO(os.getenv('MODEL_PATH', self._ensure_weights()))

    def _ensure_weights(self):
        if self.weights_url:                      # if you set a direct URL in config
            return self.weights_url
        url = sign_storage_url(self.bucket, self.path, expires=3600)
        fd, local = tempfile.mkstemp(suffix=os.path.splitext(self.path)[1]); os.close(fd)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local, "wb") as f:
                for ch in r.iter_content(1<<20):
                    if ch: f.write(ch)
        return local

    def infer(self, frame):
        # Return dict-of-arrays (xyxy, conf, cls)
        res = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
        if not res: return {"xyxy": [], "conf": [], "cls": []}
        b = res[0].boxes
        if b is None: return {"xyxy": [], "conf": [], "cls": []}
        xyxy = b.xyxy.cpu().numpy().tolist()
        conf = b.conf.cpu().numpy().tolist()
        cls  = b.cls.cpu().numpy().tolist()
        return {"xyxy": xyxy, "conf": conf, "cls": cls}
