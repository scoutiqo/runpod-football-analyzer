# pipeline.py (Updated with Skeletal Tracking + Error Handling + Basic Events)
import cv2
import mediapipe as mp  # New: For skeletal
from detector import Detector
from tracker_players import PlayerTracker
from ball_tracker import BallTracker
from calibrate import estimate_homography, image_to_field
from team_assign import TeamAssigner
from smooth import smooth_and_speed
from events import infer_possession, detect_passes  # Integrate events

def _normalize_dets(raw):
    """
    Normalize detector output to dict-of-arrays:
      {"xyxy": [[x1,y1,x2,y2], ...], "conf": [..], "cls": [..]}
    Accepts:
      - dict-of-arrays (passes through)
      - list-of-dicts with keys x1,y1,x2,y2,conf,cls/class_id/class
    """
    if isinstance(raw, dict) and "xyxy" in raw:
        return {
            "xyxy": raw.get("xyxy") or [],
            "conf": raw.get("conf") or [],
            "cls":  raw.get("cls")  or []
        }

    xyxy, conf, cls = [], [], []
    if isinstance(raw, (list, tuple)):
        for d in raw:
            if not isinstance(d, dict):
                continue
            x1 = d.get("x1"); y1 = d.get("y1"); x2 = d.get("x2"); y2 = d.get("y2")
            if None in (x1, y1, x2, y2):
                continue
            xyxy.append([float(x1), float(y1), float(x2), float(y2)])
            conf.append(float(d.get("conf", 0.0)))
            cval = d.get("cls", d.get("class_id", d.get("class", -1)))
            try:
                c = int(getattr(cval, "item", lambda: cval)())
            except:
                c = -1
            cls.append(c)
    return {"xyxy": xyxy, "conf": conf, "cls": cls}

def run_pipeline(video_path: str, cfg: dict | None = None, max_frames=150, frame_skip=2):
    """
    cfg is optional to remain backward-compatible with handler.py.
    Updated: Added MediaPipe skeletal, error handling, basic events.
    """
    cfg = cfg or {}

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
    except Exception as e:
        raise RuntimeError(f"Video open error: {str(e)}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = Detector(cfg)
    player_tracker = PlayerTracker(cfg.get("player_conf", 0.3))
    ball_tracker = BallTracker(cfg.get("ball_conf", 0.1))
    tass = TeamAssigner()

    mp_pose = mp.solutions.pose  # New: Skeletal init
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)  # 0=fast, 1=accurate, 2=slow

    Hmat = None  # homography matrix
    points = []  # raw track records
    ball_series = []  # for events
    players_by_t = {}  # t -> [players] for events

    t = 0.0
    frames = 0
    try:
        while cap.isOpened():
            for _ in range(frame_skip):
                ret, frame = cap.read()
                if not ret:
                    break
                t += 1.0 / fps

            if not ret:
                break

            # Detect
            try:
                det_raw = detector.infer(frame)
                dets = _normalize_dets(det_raw)
            except Exception as e:
                print(f"Detection error at t={t}: {str(e)}")  # Log, continue
                continue

            # Split players/ball
            player_dets = {"xyxy": [], "conf": [], "cls": []}
            ball_dets = {"xyxy": [], "conf": [], "cls": []}
            for i in range(len(dets["xyxy"])):
                if dets["cls"][i] == 0:  # player
                    player_dets["xyxy"].append(dets["xyxy"][i])
                    player_dets["conf"].append(dets["conf"][i])
                    player_dets["cls"].append(dets["cls"][i])
                elif dets["cls"][i] == 32:  # ball
                    ball_dets["xyxy"].append(dets["xyxy"][i])
                    ball_dets["conf"].append(dets["conf"][i])
                    ball_dets["cls"].append(dets["cls"][i])

            # Track
            players = player_tracker.update(player_dets)
            ball = ball_tracker.update(ball_dets)

            # Calibrate (homography) every 15 frames if not set
            if Hmat is None and frames % 15 == 0:
                keypoints = []  # Collect visible points for estimation
                for p in players:
                    keypoints.append(((p["x1"] + p["x2"]) / 2, (p["y1"] + p["y2"]) / 2))
                if ball:
                    keypoints.append(((ball[0]["x1"] + ball[0]["x2"]) / 2, (ball[0]["y1"] + ball[0]["y2"]) / 2))
                Hmat = estimate_homography(keypoints)  # Your calibrate func

            # Team assignment
            if players:
                tass.observe(
                    frame,
                    [{"id":p["id"], "x1":p["x1"], "y1":p["y1"], "x2":p["x2"], "y2":p["y2"]} for p in players]
                )

            # Build records + Skeletal
            cur_players = []  # For events
            for p in players or []:
                cx = (p["x1"] + p["x2"]) / 2.0
                cy = (p["y1"] + p["y2"]) / 2.0
                rec = {
                    "t": round(t, 3),
                    "type": "player",
                    "id": p["id"],
                    "team": tass.get_team(p["id"]),
                    "x_px": float(cx),
                    "y_px": float(cy)
                }
                xy = image_to_field(Hmat, cx, cy)
                if xy:
                    rec["x_m"], rec["y_m"] = xy

                # New: Skeletal keypoints
                try:
                    patch = frame[int(p["y1"]):int(p["y2"]), int(p["x1"]):int(p["x2"])]
                    if patch.size == 0:
                        raise ValueError("Empty patch")
                    rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb_patch)
                    if results.pose_landmarks:
                        keypoints = []
                        for lm in results.pose_landmarks.landmark:
                            # Project local (0-1) to field coords
                            lx = p["x1"] + lm.x * (p["x2"] - p["x1"])
                            ly = p["y1"] + lm.y * (p["y2"] - p["y1"])
                            lz = lm.z * 1000  # Scale depth
                            m_xy = image_to_field(Hmat, lx, ly)
                            keypoints.append({"x_m": m_xy[0] if m_xy else lx, "y_m": m_xy[1] if m_xy else ly, "z_m": lz})
                        rec["keypoints"] = keypoints  # 33 joints
                except Exception as e:
                    print(f"Skeletal error for player {p['id']} at t={t}: {str(e)}")  # Continue

                points.append(rec)
                cur_players.append(rec)

            # Ball
            if ball:
                cx = (ball[0]["x1"] + ball[0]["x2"]) / 2.0
                cy = (ball[0]["y1"] + ball[0]["y2"]) / 2.0
                rec = {
                    "t": round(t, 3),
                    "type": "ball",
                    "x_px": float(cx),
                    "y_px": float(cy)
                }
                xy = image_to_field(Hmat, cx, cy)
                if xy:
                    rec["x_m"], rec["y_m"] = xy
                points.append(rec)
                ball_series.append(rec)

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
    for e in events:
        e["possession_team"] = infer_possession(cur_players, ball_series, e["t"])

    return {
        "version": 2,
        "meta": {
            "fps": fps,
            "width": W, "height": H,
            "frame_skip": frame_skip,
            "pitch_m": [105, 68],
            "homography": (Hmat.tolist() if Hmat is not None else None)
        },
        "tracks": tracks,
        "events": events,
        "metrics": {}  # Add from evaluator.py later
    }
