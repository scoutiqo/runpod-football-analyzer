# core/pipeline.py
# Stable: player tracking + custom player/ball detector + pitch mask + smoothing

import os
import argparse
import json
import logging

import cv2
import numpy as np

try:
    import mediapipe as mp  # reserved for future skeletal work
except ImportError:  # pragma: no cover
    mp = None

# ---------------------------------------------------------------------
# Detector import / fallback
# ---------------------------------------------------------------------
try:
    # When running as "python core/pipeline.py" and detector.py is in core/
    from detector import Detector  # type: ignore
except Exception:  # pragma: no cover
    # Fallback: simple local YOLO wrapper if detector.py is not importable
    from ultralytics import YOLO  # type: ignore

    class Detector:  # type: ignore
        def __init__(self, cfg: dict | None = None):
            cfg = cfg or {}
            self.conf = float(cfg.get("conf", 0.25))
            self.iou = float(cfg.get("iou", 0.45))
            # Default to your trained weights; override via cfg["weights_path"]
            model_path = cfg.get(
                "weights_path",
                "runs/detect/train2/weights/best.pt",
            )
            self.model = YOLO(model_path)

        def infer(self, frame):
            res = self.model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )
            if not res:
                return {"xyxy": [], "conf": [], "cls": []}
            b = res[0].boxes
            if b is None:
                return {"xyxy": [], "conf": [], "cls": []}
            return {
                "xyxy": b.xyxy.cpu().numpy().tolist(),
                "conf": b.conf.cpu().numpy().tolist(),
                "cls": b.cls.cpu().numpy().tolist(),
            }

from tracker_players import PlayerTracker
from ball_tracker import BallTracker
from calibrate import estimate_homography, image_to_field
from team_assign import TeamAssigner
from smooth import smooth_and_speed


log = logging.getLogger("pipeline")

# In your Label Studio / YOLO dataset the classes are:
# 0 = ball, 1 = coach_staff, 2 = goalkeeper, 3 = player, 4 = referee
PLAYER_CLASS_IDS = {3}
BALL_CLASS_IDS = {0}


def infer_possession(*args, **kwargs):
    """Stub – will be filled later."""
    return {}


def detect_passes(*args, **kwargs):
    """Stub – will be filled later."""
    return []


# ---------------------------------------------------------------------
# Pitch mask (keep only grass area)
# ---------------------------------------------------------------------
def compute_pitch_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # generous green range; adjust if your grass is different
    lower = np.array([35, 40, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    m = cv2.inRange(hsv, lower, upper)
    m = cv2.medianBlur(m, 7)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m
    big = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(m)
    cv2.drawContours(mask, [big], -1, 255, thickness=cv2.FILLED)
    return mask


def inside_pitch(x: float, y: float, mask: np.ndarray) -> bool:
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or yi >= mask.shape[0] or xi >= mask.shape[1]:
        return False
    return mask[yi, xi] > 0


# ---------------------------------------------------------------------
# Ball fallback scoring
# ---------------------------------------------------------------------
def _ball_color_score(patch_bgr: np.ndarray) -> float:
    """Score 0..1 for white/orange-like ball colors."""
    if patch_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = ((S < 60) & (V > 170)).mean()
    orange = (((H < 20) | (H > 160)) & (S > 90) & (V > 100)).mean()
    return float(max(white, orange))


def _aspect_roundness(w: float, h: float) -> float:
    if w <= 0 or h <= 0:
        return 0.0
    return float(min(w, h) / max(w, h))  # 1 = square-ish


def pick_ball_candidate(
    frame: np.ndarray,
    candidates,
    last_xy: tuple[float, float] | None,
    W: int,
    H: int,
):
    """
    candidates: list of (x1,y1,x2,y2, conf, cls)
    last_xy: (x,y) or None
    Return best [x1,y1,x2,y2] or None
    """
    if not candidates:
        return None

    min_side = 0.012 * min(W, H)
    max_side = 0.080 * min(W, H)

    best_score, best_box = -1e9, None
    for (x1, y1, x2, y2, conf, cls_) in candidates:
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        side_ok = (min_side <= max(w, h) <= max_side)

        x1i = int(max(0, min(x1, frame.shape[1] - 1)))
        y1i = int(max(0, min(y1, frame.shape[0] - 1)))
        x2i = int(max(0, min(x2, frame.shape[1] - 1)))
        y2i = int(max(0, min(y2, frame.shape[0] - 1)))
        if x2i <= x1i or y2i <= y1i:
            continue
        patch = frame[y1i:y2i, x1i:x2i]

        color = _ball_color_score(patch)
        roundness = _aspect_roundness(w, h)

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist = 0.0 if last_xy is None else (
            np.hypot(cx - last_xy[0], cy - last_xy[1]) / float(min(W, H))
        )

        # Weighted score: color + roundness + plausible size + ball class hint + conf - distance from last ball
        score = (
            2.5 * color
            + 1.5 * roundness
            + (0.6 if side_ok else 0.0)
            + 0.4 * float(cls_ in BALL_CLASS_IDS)
            + 0.4 * conf
            - 0.5 * dist
        )
        if score > best_score:
            best_score, best_box = score, [x1, y1, x2, y2]
    return best_box


# ---------------------------------------------------------------------
# Detection normalization
# ---------------------------------------------------------------------
def _normalize_dets(raw):
    """Normalize detector output to dict-of-arrays."""
    if isinstance(raw, dict) and "xyxy" in raw:
        return {
            "xyxy": raw.get("xyxy") or [],
            "conf": raw.get("conf") or [],
            "cls": raw.get("cls") or [],
        }

    xyxy, conf, cls = [], [], []
    if isinstance(raw, (list, tuple)):
        for d in raw:
            if not isinstance(d, dict):
                continue
            x1 = d.get("x1")
            y1 = d.get("y1")
            x2 = d.get("x2")
            y2 = d.get("y2")
            if None in (x1, y1, x2, y2):
                continue
            xyxy.append([float(x1), float(y1), float(x2), float(y2)])
            conf.append(float(d.get("conf", 0.0)))
            cval = d.get("cls", d.get("class_id", d.get("class", -1)))
            try:
                c = int(getattr(cval, "item", lambda: cval)())
            except Exception:
                c = -1
            cls.append(c)
    return {"xyxy": xyxy, "conf": conf, "cls": cls}


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def run_pipeline(
    video_path: str,
    cfg: dict | None = None,
    max_frames: int = 150,
    frame_skip: int = 2,
):
    """Stable pipeline: players + ball + pitch mask + smoothing."""

    cfg = cfg or {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector_cfg = cfg.get("detector", {})
    tracking_cfg = cfg.get("tracking", {"max_age": 30, "min_hits": 3})
    ball_cfg = cfg.get("ball", {"min_conf": 0.10})

    # Force your custom weights path if not provided
    detector_cfg.setdefault("weights_path", "runs/detect/train2/weights/best.pt")

    detector = Detector(detector_cfg)
    player_tracker = PlayerTracker(tracking_cfg)
    ball_tracker = BallTracker(ball_cfg)
    tass = TeamAssigner()

    Hmat = None
    points = []          # output track points (players + ball)
    ball_series = []     # for events
    players_by_t = {}    # t -> [player recs]

    pitch_mask = None        # keep-only-grass mask (computed from the frame)
    last_ball_xy: tuple[float, float] | None = None

    t = 0.0
    frames = 0

    while cap.isOpened():
        # frame skipping and max_frames control
        ret = True
        for _ in range(frame_skip):
            if frames >= max_frames:
                ret = False
                break
            ret, frame = cap.read()
            if not ret:
                break
            t += 1.0 / fps
            frames += 1

        if not ret:
            break

        # Refresh pitch mask first frame and every ~10s
        if pitch_mask is None or (frames % 300 == 0):
            pitch_mask = compute_pitch_mask(frame)

        # Detect
        try:
            det_raw = detector.infer(frame)
            dets = _normalize_dets(det_raw)
        except Exception as e:  # pragma: no cover
            log.error("Detection error at t=%.3f: %s", t, e)
            continue

        # Split players/ball + robust fallback that ALWAYS picks a ball candidate
        player_dets = {"xyxy": [], "conf": [], "cls": []}
        ball_dets = {"xyxy": [], "conf": [], "cls": []}
        candidates = []  # (area, conf, box, cls)

        for i in range(len(dets["xyxy"])):
            x1, y1, x2, y2 = dets["xyxy"][i]
            conf = float(dets["conf"][i])
            cls_raw = dets["cls"][i]
            cls = int(cls_raw) if isinstance(cls_raw, (int, float)) else -1

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h  # (kept for potential future use)

            # Drop detections whose centers are off-pitch
            cx_det, cy_det = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if pitch_mask is not None and not inside_pitch(cx_det, cy_det, pitch_mask):
                continue

            candidates.append((area, conf, [x1, y1, x2, y2], cls))

            if cls in PLAYER_CLASS_IDS:
                player_dets["xyxy"].append([x1, y1, x2, y2])
                player_dets["conf"].append(conf)
                player_dets["cls"].append(cls)

            if cls in BALL_CLASS_IDS:
                ball_dets["xyxy"].append([x1, y1, x2, y2])
                ball_dets["conf"].append(conf)
                ball_dets["cls"].append(cls)

        # Ball fallback: if no ball box but we have candidates, pick one
        if not ball_dets["xyxy"] and candidates:
            cand2 = []
            for (_a, c, box, cl) in candidates:
                x1, y1, x2, y2 = box
                cand2.append((x1, y1, x2, y2, c, cl))
            pick = pick_ball_candidate(frame, cand2, last_ball_xy, W, H)
            if pick is not None:
                x1, y1, x2, y2 = pick
                ball_dets["xyxy"].append([x1, y1, x2, y2])
                ball_dets["conf"].append(0.5)  # synthetic conf for fallback
                ball_dets["cls"].append(next(iter(BALL_CLASS_IDS)))

        # Track
        players = player_tracker.update(player_dets)
        ball = ball_tracker.update(ball_dets)

        # Normalize ball (list/dict/empty) and provide fallback box
        ball_box = None
        if isinstance(ball, list) and ball and isinstance(ball[0], dict) and all(
            k in ball[0] for k in ("x1", "y1", "x2", "y2")
        ):
            ball_box = ball[0]
        elif isinstance(ball, dict) and all(k in ball for k in ("x1", "y1", "x2", "y2")):
            ball_box = ball
        elif ball_dets["xyxy"]:
            x1, y1, x2, y2 = ball_dets["xyxy"][0]
            ball_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        # Calibrate (homography) occasionally
        if Hmat is None and frames % 15 == 0:
            keypoints = []
            for p in (players or []):
                keypoints.append(((p["x1"] + p["x2"]) / 2, (p["y1"] + p["y2"]) / 2))
            if ball_box:
                keypoints.append(
                    ((ball_box["x1"] + ball_box["x2"]) / 2, (ball_box["y1"] + ball_box["y2"]) / 2)
                )
            if len(keypoints) >= 4:
                try:
                    Hmat = estimate_homography(keypoints)
                except Exception:
                    Hmat = None

        # Team assignment
        if players:
            tass.observe(
                frame,
                [
                    {"id": p["id"], "x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"]}
                    for p in players
                ],
            )

        # Players -> points
        cur_players = []
        for p in (players or []):
            cx = (p["x1"] + p["x2"]) / 2.0
            cy = (p["y1"] + p["y2"]) / 2.0
            rec = {
                "t": round(t, 3),
                "type": "player",
                "id": p["id"],
                "team": tass.get_team(p["id"]),
                "x_px": float(cx),
                "y_px": float(cy),
            }
            xy = image_to_field(Hmat, cx, cy)
            if xy:
                rec["x_m"], rec["y_m"] = xy

            points.append(rec)
            cur_players.append(rec)

        # Ball -> points (use normalized tracker output or fallback)
        if ball_box:
            cx = (ball_box["x1"] + ball_box["x2"]) / 2.0
            cy = (ball_box["y1"] + ball_box["y2"]) / 2.0
            rec = {"t": round(t, 3), "type": "ball", "x_px": float(cx), "y_px": float(cy)}
            xy = image_to_field(Hmat, cx, cy)
            if xy:
                rec["x_m"], rec["y_m"] = xy
            points.append(rec)
            ball_series.append(rec)
            last_ball_xy = (cx, cy)

        # For events
        players_by_t[t] = cur_players

    cap.release()

    # Smooth & speed
    tracks = smooth_and_speed(points, have_metric=(Hmat is not None))

    # Events (basic)
    events = detect_passes(ball_series, players_by_t)

    # Possession per event: use closest player set in time
    def _players_at(time_q):
        if not players_by_t:
            return []
        key = min(players_by_t.keys(), key=lambda tt: abs(tt - time_q))
        return players_by_t.get(key, [])

    for e in events:
        e["possession_team"] = infer_possession(_players_at(e["t"]), ball_series, e["t"])

    return {
        "version": 2,
        "meta": {
            "fps": fps,
            "width": W,
            "height": H,
            "frame_skip": frame_skip,
            "pitch_m": [105, 68],
            "homography": (Hmat.tolist() if Hmat is not None else None),
        },
        "tracks": tracks,
        "events": events,
        "metrics": {},
    }


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("--max_frames", type=int, default=6000)
    ap.add_argument("--frame_skip", type=int, default=3)
    ap.add_argument("--out", default="tracks.json", help="Where to save output JSON")
    ap.add_argument(
        "--weights",
        default="runs/detect/train2/weights/best.pt",
        help="Path to custom YOLO weights (players_v1_best.pt)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    args = _parse_args()

    log.info(
        "Starting: video=%s max_frames=%s frame_skip=%s weights=%s",
        args.video,
        args.max_frames,
        args.frame_skip,
        args.weights,
    )

    cfg = {
        "detector": {
            "weights_path": args.weights,
            "conf": 0.25,
            "iou": 0.45,
        },
        "tracking": {"max_age": 30, "min_hits": 3},
        "ball": {"min_conf": 0.05},
    }

    try:
        out = run_pipeline(
            args.video,
            cfg=cfg,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
        )
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f)
        size_kb = os.path.getsize(args.out) / 1024.0
        log.info("Saved %s (%.1f KB)", args.out, size_kb)
    except Exception as e:  # pragma: no cover
        log.exception("Pipeline crashed: %s", e)
        raise
