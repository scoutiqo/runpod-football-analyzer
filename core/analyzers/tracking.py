# analyzers/tracking.py
from __future__ import annotations
import json, math, os, cv2, numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import os
from ultralytics import YOLO
import supervision as sv

from analyzers.ball_tracker import BallTracker

PERSON_ID = 0         # COCO 'person'
SPORTS_BALL_ID = 32   # COCO 'sports ball'

@dataclass
class TrackerCfg:
    yolo_model: str = "yolov8x.pt"  # players + sports ball
    conf: float = 0.25
    iou: float = 0.5
    max_cosine_dist: float = 0.2     # not used by ByteTrack; kept for future DeepSORT
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 90
    nms: float = 0.7
    ball_gap_flow: int = 20
    draw_trails: int = 30

def _norm(cx, cy, w, h):
    return cx / float(w), cy / float(h)

def _video_writer(out_path: Path, w: int, h: int, fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

def run_tracking(
    video_path: str,
    out_dir: str,
    cfg: TrackerCfg = TrackerCfg(),
) -> Tuple[str, str]:
    """
    Returns: (tracks_json_path, overlay_mp4_path)
    """
    out_dir_p = Path(out_dir); out_dir_p.mkdir(parents=True, exist_ok=True)
    tracks_json = out_dir_p / "tracks.json"
    overlay_mp4 = out_dir_p / "overlay.mp4"

    model = YOLO(os.getenv('MODEL_PATH', cfg.yolo_model))

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else None

    writer = _video_writer(overlay_mp4, W, H, fps)

    # Player tracker
    bytetrack = sv.ByteTrack(track_thresh=cfg.track_thresh,
                             match_thresh=cfg.match_thresh,
                             track_buffer=cfg.track_buffer)

    # Annotators
    box_anno   = sv.BoxAnnotator(thickness=2)
    label_anno = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    trace_anno = sv.TraceAnnotator(thickness=2, trace_length=cfg.draw_trails)

    # Ball
    ball = BallTracker(max_flow_gap=cfg.ball_gap_flow)

    # Storage
    player_tracks: Dict[int, List[Tuple[int, float, float, float, float]]] = {}  # tid -> [(f, cxn, cyn, wn, hn)]
    ball_track: List[Tuple[int, float, float]] = []  # [(f, cxn, cyn)]
    trails: Dict[int, List[Tuple[int,int]]] = {}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        # YOLO inference
        res = model.predict(source=frame, conf=cfg.conf, iou=cfg.iou, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)

        # Split classes
        person_mask = det.class_id == PERSON_ID if det.class_id is not None else np.array([], dtype=bool)
        ball_mask   = det.class_id == SPORTS_BALL_ID if det.class_id is not None else np.array([], dtype=bool)

        # PLAYERS → ByteTrack
        person_det = det[person_mask] if len(det) else det
        tracks = bytetrack.update_with_detections(person_det)

        # Annotate players
        player_labels = []
        if len(tracks) > 0:
            # save normalized centers per player
            for tid, bbox in zip(tracks.tracker_id, tracks.xyxy):
                x1,y1,x2,y2 = bbox
                cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                cxn, cyn = _norm(cx, cy, W, H)
                wn, hn   = _norm(x2-x1, y2-y1, W, H)
                player_tracks.setdefault(int(tid), []).append((frame_idx, cxn, cyn, wn, hn))
                trails.setdefault(int(tid), []).append((int(cx), int(cy)))
                player_labels.append(f"{int(tid)}")

            frame = box_anno.annotate(scene=frame, detections=tracks)
            frame = label_anno.annotate(scene=frame, detections=tracks, labels=player_labels)
            frame = trace_anno.annotate(scene=frame, detections=tracks)

        # BALL → prefer detection; else optical flow
        ball_updated = False
        if len(det) and ball_mask.any():
            # choose highest-confidence ball
            idxs = np.where(ball_mask)[0]
            if len(idxs):
                best = int(idxs[np.argmax(det.confidence[idxs])])
                bbox = det.xyxy[best]
                ball.update_with_detection(frame, bbox)
                cx, cy = ball.current_xy()
                cxn, cyn = _norm(cx, cy, W, H)
                ball_track.append((frame_idx, cxn, cyn))
                cv2.circle(frame, (int(cx), int(cy)), 7, (0,255,255), -1)
                ball_updated = True

        if not ball_updated:
            of = ball.update_optical_flow(frame)
            if of is not None:
                cx, cy = of
                cxn, cyn = _norm(cx, cy, W, H)
                ball_track.append((frame_idx, cxn, cyn))
                cv2.circle(frame, (int(cx), int(cy)), 7, (0,255,255), -1)

        writer.write(frame)
        frame_idx += 1

    writer.release(); cap.release()

    # Build JSON result (minimal, frame-wise points)
    out = {
        "video": {"fps": fps, "width": W, "height": H, "frames": total_frames},
        "players": [
            {
                "tid": int(tid),
                "frames": [{"f": f, "cx": cx, "cy": cy, "w": w, "h": h} for (f,cx,cy,w,h) in seq]
            }
            for tid, seq in sorted(player_tracks.items(), key=lambda x: x[0])
        ],
        "ball": [{"f": f, "cx": cx, "cy": cy} for (f,cx,cy) in ball_track]
    }
    with open(tracks_json, "w", encoding="utf-8") as f:
        json.dump(out, f)

    return str(tracks_json), str(overlay_mp4)