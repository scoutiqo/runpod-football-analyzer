# core/detector.py
# Simple local YOLO detector: no config, no Supabase, no sign_storage_url

from ultralytics import YOLO


class Detector:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        # If cfg provides a path, use it; otherwise default to COCO or your custom model
        self.weights_path = cfg.get("weights_path", "custom_players_v1.pt")
        self.conf = float(cfg.get("conf", 0.25))
        self.iou = float(cfg.get("iou", 0.45))

        # This loads weights from a local .pt file
        self.model = YOLO(self.weights_path)

    def infer(self, frame):
        """
        Run YOLO on a single frame.
        Returns a dict with 'xyxy', 'conf', 'cls' lists.
        """
        res = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
        if not res:
            return {"xyxy": [], "conf": [], "cls": []}

        b = res[0].boxes
        if b is None:
            return {"xyxy": [], "conf": [], "cls": []}

        xyxy = b.xyxy.cpu().numpy().tolist()
        conf = b.conf.cpu().numpy().tolist()
        cls = b.cls.cpu().numpy().tolist()

        return {"xyxy": xyxy, "conf": conf, "cls": cls}
