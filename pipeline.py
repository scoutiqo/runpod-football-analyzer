# pipeline.py (Stable: player tracking + robust ball fallback + skeletal + events + pitch mask + sticky ball)
import cv2
import mediapipe as mp
import numpy as np

from detector import Detector
from tracker_players import PlayerTracker
from ball_tracker import BallTracker
from calibrate import estimate_homography, image_to_field
from team_assign import TeamAssigner
from smooth import smooth_and_speed
from events import infer_possession, detect_passes


# ---------- Pitch mask (keep only grass area) ----------
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


# ---------- Ball fallback scoring ----------
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


def pick_ball_candidate(frame: np.ndarray, candidates, last_xy, W, H):
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

        x1i, y1i = int(max(0, min(x1, frame.shape[1] - 1))), int(max(0, min(y1, frame.shape[0] - 1)))
        x2i, y2i = int(max(0, min(x2, frame.shape[1] - 1))), int(max(0, min(y2, frame.shape[0] - 1)))
        if x2i <= x1i or y2i <= y1i:
            continue
        patch = frame[y1i:y2i, x1i:x2i]

        color = _ball_color_score(patch)
        roundness = _aspect_roundness(w, h)

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist = 0.0 if last_xy is None else (np.hypot(cx - last_xy[0], cy - last_xy[1]) / float(min(W, H)))

        # Weighted score: color + roundness + plausible size + coco class hint + conf - distance from last ball
        score = 2.5 * color + 1.5 * roundness + (0.6 if side_ok else 0.0) + 0.4 * float(cls_ == 32) + 0.4 * conf - 0.5 * dist
        if score > best_score:
            best_score, best_box = score, [x1, y1, x2, y2]
    return best_box


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


def run_pipeline(video_path: str, cfg: dict | None = None, max_frames=150, frame_skip=2):
    """Adds robust ball fallback so ball points are always produced."""
    cfg = cfg or {}

    # Open video
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
    except Exception as e:
        raise RuntimeError(f"Video open error: {str(e)}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Config split
    detector_cfg = cfg.get("detector", {})
    tracking_cfg = cfg.get("tracking", {"max_age": 30, "min_hits": 3})
    ball_cfg = cfg.get("ball", {"min_conf": 0.10, "class_id": 32})

    detector = Detector(detector_cfg)
    player_tracker = PlayerTracker(tracking_cfg)
    ball_tracker = BallTracker(ball_cfg)
    tass = TeamAssigner()

    # Skeletal
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)

    Hmat = None
    points = []          # output track points (players + ball)
    ball_series = []     # for events
    players_by_t = {}    # t -> [player recs]

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # EXACTLY HERE: init the new state variables
    pitch_mask = None        # keep-only-grass mask (computed from the frame)
    last_ball_xy = None      # last accepted ball center (x,y) to stabilize fallback
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    t = 0.0
    frames = 0
    try:
        while cap.isOpened():
            # advance by frame_skip
            for _ in range(frame_skip):
                ret, frame = cap.read()
                if not ret:
                    break
                t += 1.0 / fps
            if not ret:
                break

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # EXACTLY HERE: refresh pitch mask first frame and every ~10s
            if pitch_mask is None or (frames % 300 == 0):
                pitch_mask = compute_pitch_mask(frame)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # Detect
            try:
                det_raw = detector.infer(frame)
                dets = _normalize_dets(det_raw)
            except Exception as e:
                print(f"Detection error at t={t}: {str(e)}")
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
                area = w * h

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # EXACTLY HERE: drop detections whose centers are off-pitch
                cx_det, cy_det = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if pitch_mask is not None and not inside_pitch(cx_det, cy_det, pitch_mask):
                    continue
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                candidates.append((area, conf, [x1, y1, x2, y2], cls))

                if cls == 0:  # person
                    player_dets["xyxy"].append([x1, y1, x2, y2])
                    player_dets["conf"].append(conf)
                    player_dets["cls"].append(cls)

                if cls == 32:  # sports ball
                    ball_dets["xyxy"].append([x1, y1, x2, y2])
                    ball_dets["conf"].append(conf)
                    ball_dets["cls"].append(cls)

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # EXACTLY HERE: replace previous "pick smallest" with scored fallback
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
                    ball_dets["cls"].append(32)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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

            # Players -> points (with skeletal)
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

                # Skeletal
                try:
                    patch = frame[int(p["y1"]): int(p["y2"]), int(p["x1"]): int(p["x2"])]
                    if patch.size > 0:
                        rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                        results = pose.process(rgb_patch)
                        if results.pose_landmarks:
                            kps = []
                            for lm in results.pose_landmarks.landmark:
                                lx = p["x1"] + lm.x * (p["x2"] - p["x1"])
                                ly = p["y1"] + lm.y * (p["y2"] - p["y1"])
                                lz = lm.z * 1000
                                m_xy = image_to_field(Hmat, lx, ly)
                                kps.append(
                                    {"x_m": m_xy[0] if m_xy else lx, "y_m": m_xy[1] if m_xy else ly, "z_m": lz}
                                )
                            rec["keypoints"] = kps
                except Exception as e:
                    print(f"Skeletal error for player {p.get('id')} at t={t}: {str(e)}")

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
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # EXACTLY HERE: remember last ball center to stabilize fallback
                last_ball_xy = (cx, cy)
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # For events
            players_by_t[t] = cur_players

            frames += 1
            if frames >= max_frames:
                break
    finally:
        cap.release()

    # Smooth
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


if __name__ == "__main__":
    import os
    import argparse
    import logging
    import json
    import config

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    log = logging.getLogger("pipeline")

    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max_frames", type=int, default=6000)
    ap.add_argument("--frame_skip", type=int, default=3)
    ap.add_argument("--out", default="tracks.json", help="Where to save output JSON")
    args = ap.parse_args()

    log.info(
        "Starting: video=%s max_frames=%s frame_skip=%s",
        args.video,
        args.max_frames,
        args.frame_skip,
    )

    # Fetch config and force COCO weights so class 32 (sports ball) exists
    try:
        cfg = config.fetch_config("default")
        cfg.setdefault("detector", {})
        cfg["detector"]["weights_bucket"] = "local"
        cfg["detector"]["weights_path"] = "yolov8n.pt"
    except Exception as e:
        log.warning("fetch_config failed (%s); using local yolov8n.pt", e)
        cfg = {"detector": {"weights_bucket": "local", "weights_path": "yolov8n.pt"}}

    # Run and save output
    try:
        out = run_pipeline(args.video, cfg=cfg, max_frames=args.max_frames, frame_skip=args.frame_skip)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f)
        log.info("Saved %s (%.1f KB)", args.out, os.path.getsize(args.out) / 1024.0)
    except Exception as e:
        log.exception("Pipeline crashed: %s", e)
        raise
