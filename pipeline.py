import cv2
from detector import Detector
from tracker_players import PlayerTracker
from ball_tracker import BallTracker
from calibrate import estimate_homography, image_to_field
from team_assign import TeamAssigner
from smooth import smooth_and_speed

def run_pipeline(video_path: str, max_frames=150, frame_skip=2):
    cap = cv2.VideoCapture(video_path); assert cap.isOpened(), "OpenCV failed to open video"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # read first frame for homography attempt
    ok, first = cap.read(); assert ok
    Hmat = estimate_homography(first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    det  = Detector()
    ptrk = PlayerTracker()
    btrk = BallTracker()
    tass = TeamAssigner()

    raw = []
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
# Normalize detector output so trackers never see None
dets = det.infer(frame) or {}
if not isinstance(dets, dict):
    dets = {}
dets.setdefault("xyxy", [])
dets.setdefault("conf", [])
dets.setdefault("cls",  [])

players = ptrk.update(dets)      # [{id,x1,y1,x2,y2,cls,conf}, ...]
ball    = btrk.update(dets)      # {x1,y1,x2,y2,cls,conf} or None


        # update team model with current boxes
        tass.observe(frame, [{"id":p["id"], "x1":p["x1"], "y1":p["y1"], "x2":p["x2"], "y2":p["y2"]} for p in players])

        # players -> centers (+team)
        for p in players:
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
            raw.append(rec)

        # ball
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
            raw.append(rec)

        frames += 1
        if frames >= max_frames:
            break

    cap.release()

    # smooth + speed
    have_metric = Hmat is not None
    tracks = smooth_and_speed(raw, have_metric=have_metric)

    result = {
        "version": 2,
        "meta": {
            "fps": fps,
            "width": W, "height": H,
            "frame_skip": frame_skip,
            "pitch_m": [105, 68],
            "homography": (Hmat.tolist() if Hmat is not None else None)
        },
        "tracks": tracks,
        "events": [],      # next milestones
        "metrics": {}      # next milestones
    }
    return result
