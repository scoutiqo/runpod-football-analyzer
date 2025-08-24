import cv2
from detector import Detector
from tracker_players import PlayerTracker
from ball_tracker import BallTracker
from calibrate import estimate_homography, image_to_field

def run_pipeline(video_path: str, max_frames=150, frame_skip=2):
    cap = cv2.VideoCapture(video_path); assert cap.isOpened(), "OpenCV failed to open video"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # read first frame for homography attempt
    ok, first = cap.read(); assert ok
    Hmat = estimate_homography(first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    det = Detector()
    ptrk = PlayerTracker()
    btrk = BallTracker()

    tracks = []
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
        dets = det.infer(frame)
        players = ptrk.update(dets)
        ball = btrk.update(dets)

        # players -> centers
        for p in players:
            cx = (p["x1"] + p["x2"]) / 2.0
            cy = (p["y1"] + p["y2"]) / 2.0
            rec = {
                "t": round(t, 3),
                "type": "player",
                "id": p["id"],
                "x_px": float(cx),
                "y_px": float(cy)
            }
            xy = image_to_field(Hmat, cx, cy)
            if xy:
                rec["x_m"], rec["y_m"] = xy
            tracks.append(rec)

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
            tracks.append(rec)

        frames += 1
        if frames >= max_frames:
            break

    cap.release()

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
        "events": [],      # filled in next milestones
        "metrics": {}      # filled in next milestones
    }
    return result
