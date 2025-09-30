from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

# COCO IDs
PLAYER_COCO_ID = 0   # person
BALL_COCO_ID   = 32  # sports ball

class Tracker:
    def __init__(self, player_model="yolov8n.pt", ball_model="yolov8n.pt"):
        # lightweight defaults; swap with your fine-tuned weights when ready
        self.player_model = YOLO(player_model)
        self.ball_model   = YOLO(ball_model)
        self.player_tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    # ---------- Players ----------
    def detect_players(self, frame):
        r = self.player_model(frame, imgsz=1280, verbose=False)[0]
        det = sv.Detections.from_ultralytics(r)
        det = det[det.class_id == PLAYER_COCO_ID]
        return det

    def track_players(self, frame):
        det = self.detect_players(frame)
        tracked = self.player_tracker.update_with_detections(det)
        return tracked

    # ---------- Ball ----------
    def detect_ball(self, frame):
        # Low-memory tiled inference to find the tiny ball
        def _cb(image_slice):
            out = self.ball_model(image_slice, imgsz=640, verbose=False)[0]
            d = sv.Detections.from_ultralytics(out)
            d = d[d.class_id == BALL_COCO_ID]
            return d

        slicer = sv.InferenceSlicer(
            callback=_cb,
            slice_wh=(480, 480),
            overlap_ratio_wh=(0, 0),
            overlap_filter_strategy="NONE",
            thread_workers=1
        )
        det = slicer(frame).with_nms(threshold=0.1)
        return det

    # ---------- Drawing ----------
    def draw(self, frame, players: sv.Detections, ball: sv.Detections | None,
             team_ids=None, possession_tid=None):
        out = frame.copy()

        ellipse_anno = sv.EllipseAnnotator(thickness=2)
        label_anno   = sv.LabelAnnotator(
            text_position=sv.Position.BOTTOM_CENTER,
            text_thickness=1
        )

        # Build labels robustly (no truthiness checks on numpy arrays)
        tids = []
        if getattr(players, "tracker_id", None) is not None:
            tids = players.tracker_id.tolist()
        labels = [str(int(t)) if t is not None else "" for t in tids]

        out = ellipse_anno.annotate(out, players)
        out = label_anno.annotate(out, players, labels=labels)

        # Ball marker (green triangle)
        if ball is not None and len(ball) > 0:
            tri = sv.TriangleAnnotator(base=18, height=14)
            out = tri.annotate(out, ball)

        # Optional: draw possession owner (simple HUD)
        if possession_tid is not None:
            cv2.putText(
                out,
                f"Possession: Team {possession_tid}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA
            )

        return out


