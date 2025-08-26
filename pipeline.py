# pipeline.py
import cv2
from detector import Detector
from tracker_players import PlayerTracker
from ball_tracker import BallTracker
from calibrate import estimate_homography, image_to_field
from team_assign import TeamAssigner
from smooth import smooth_and_speed

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
            except Exception:
                c = -1
            cls.append(c)
    return {"xyxy": xyxy, "conf": conf, "cls": cls}

def run_pipeline(video_path: str, cfg: dict | None = None, max_frames=150, frame_skip=2):
    """
    cfg is optional to remain backward-compatible with handler.py.
    """
    cfg = cfg or {}

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "OpenCV failed to open video"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # read first frame for homography attempt
    ok, first = cap.read(); assert ok, "Failed to read first frame"
    Hmat = estimate_homography(first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # allow cfg overrides
    frame_cfg   = cfg.get("frame", {})
    frame_skip  = int(frame_cfg.get("frame_skip", frame_skip))
    max_frames  = int(frame_cfg.get("max_frames", max_frames))

    det  = Detector(cfg)
    ptrk = PlayerTracker(cfg.get("tracking", {}))
    btrk = BallTracker(cfg.get("ball", {}))
    tass = TeamAssigner()

    points = []
    frames = 0
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % frame_skip != 0:
            continue

        t = idx / fps

        # detector -> normalized schema
        dets = _normalize_dets(det.infer(frame))
        # sanity assert (helps catch weird detector outputs fast)
        assert isinstance(dets, dict) and all(k in dets for k in ("xyxy","conf","cls")), "Detector output schema invalid"

        players = ptrk.update(dets)      # [{id,x1,y1,x2,y2,cls,conf}, ...]
        ball    = btrk.update(dets)      # {x1,y1,x2,y2,cls,conf} or None

        # update team model
        if players:
            tass.observe(
                frame,
                [{"id":p["id"], "x1":p["x1"], "y1":p["y1"], "x2":p["x2"], "y2":p["y2"]} for p in players]
            )

        # players -> centers (+team)
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
            points.append(rec)

        # ball center
        if ball:
            cx = (ball["x1"] + ball["x2"]) / 2.0
            cy = (ball["y1"] + ball["y2"]) / 2.0
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

        frames += 1
        if frames >= max_frames:
            break

    cap.release()

    tracks = smooth_and_speed(points, have_metric=(Hmat is not None))
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
        "events": [],
        "metrics": {}
    }
