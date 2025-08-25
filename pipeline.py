import cv2
from detector import Detector
from tracker_players import PlayerTracker
from ball_tracker import BallTracker
from calibrate import estimate_homography, image_to_field
from team_assign import TeamAssigner
from smooth import smooth_and_speed

def _normalize_dets(raw):
    # returns dict-of-arrays {xyxy, conf, cls}
    if isinstance(raw, dict) and "xyxy" in raw:
        return {
            "xyxy": raw.get("xyxy") or [],
            "conf": raw.get("conf") or [],
            "cls":  raw.get("cls")  or []
        }
    xyxy, conf, cls = [], [], []
    if isinstance(raw, (list, tuple)):
        for d in raw:
            if not isinstance(d, dict): continue
            x1=d.get("x1"); y1=d.get("y1"); x2=d.get("x2"); y2=d.get("y2")
            if None in (x1,y1,x2,y2): continue
            xyxy.append([float(x1),float(y1),float(x2),float(y2)])
            conf.append(float(d.get("conf",0.0)))
            cls.append(int(d.get("cls", d.get("class_id", -1))))
    return {"xyxy": xyxy, "conf": conf, "cls": cls}

def run_pipeline(video_path: str, cfg: dict, max_frames=150, frame_skip=2):
    cap = cv2.VideoCapture(video_path); assert cap.isOpened(), "OpenCV failed to open video"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    ok, first = cap.read(); assert ok
    Hmat = estimate_homography(first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # read frame sampling from config if present
    frame_skip = int(cfg.get("frame", {}).get("frame_skip", frame_skip))
    max_frames = int(cfg.get("frame", {}).get("max_frames", max_frames))

    det  = Detector(cfg)
    ptrk = PlayerTracker(cfg.get("tracking", {}))
    btrk = BallTracker(cfg.get("ball", {}))
    tass = TeamAssigner()

    raw_points = []
    frames = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        idx += 1
        if idx % frame_skip != 0: continue

        t = idx / fps

        dets = _normalize_dets(det.infer(frame))
        players = ptrk.update(dets)
        ball    = btrk.update(dets)

        tass.observe(frame, [{"id":p["id"],"x1":p["x1"],"y1":p["y1"],"x2":p["x2"],"y2":p["y2"]} for p in players])

        for p in players:
            cx = (p["x1"] + p["x2"]) / 2.0
            cy = (p["y1"] + p["y2"]) / 2.0
            rec = {"t": round(t, 3), "type": "player", "id": p["id"],
                   "team": tass.get_team(p["id"]), "x_px": float(cx), "y_px": float(cy)}
            xy = image_to_field(Hmat, cx, cy)
            if xy: rec["x_m"], rec["y_m"] = xy
            raw_points.append(rec)

        if ball:
            cx = (ball["x1"] + ball["x2"]) / 2.0
            cy = (ball["y1"] + ball["y2"]) / 2.0
            rec = {"t": round(t, 3), "type": "ball", "x_px": float(cx), "y_px": float(cy)}
            xy = image_to_field(Hmat, cx, cy)
            if xy: rec["x_m"], rec["y_m"] = xy
            raw_points.append(rec)

        frames += 1
        if frames >= max_frames: break

    cap.release()
    tracks = smooth_and_speed(raw_points, have_metric=(Hmat is not None))

    return {
        "version": 2,
        "meta": {"fps": fps, "width": W, "height": H, "frame_skip": frame_skip,
                 "pitch_m": [105,68], "homography": (Hmat.tolist() if Hmat is not None else None)},
        "tracks": tracks, "events": [], "metrics": {}
    }
