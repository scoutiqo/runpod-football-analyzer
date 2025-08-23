# pipeline.py
import cv2, numpy as np, json, os, uuid
from .detector import Detector
from .tracker  import PlayerTracker
from .ball     import BallTracker
from .calibrate import estimate_homography, image_to_field
from .smooth import smooth_tracks_kalman
from .shape import team_shape
# ... import events, passnet, pitchctl, metrics, vis

def run_pipeline(video_path, max_frames=150, frame_skip=2):
    cap = cv2.VideoCapture(video_path); assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    det  = Detector()
    trk  = PlayerTracker()
    btrk = BallTracker()

    # get homography attempt from first frame
    ok, first = cap.read(); assert ok
    Hmat = estimate_homography(first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracks=[]; ball_series=[]; n=0; out_idx=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        n+=1
        if n % frame_skip != 0: continue
        t = n/fps

        dets = det.infer(frame)           # raw dets
        pl   = trk.update(dets)           # player tracks
        ball = btrk.update(dets)          # ball

        # to centers + meters if homography present
        for p in pl:
            cx = (p["x1"]+p["x2"])/2; cy=(p["y1"]+p["y2"])/2
            rec = {"t":t, "type":"player","id":p["id"], "x_px":cx,"y_px":cy}
            if Hmat is not None:
                xy = image_to_field(Hmat, cx, cy)
                if xy: rec["x_m"], rec["y_m"] = xy
            tracks.append(rec)

        if ball:
            cx=(ball["x1"]+ball["x2"])/2; cy=(ball["y1"]+ball["y2"])/2
            rec={"t":t,"type":"ball","x_px":cx,"y_px":cy}
            if Hmat is not None:
                xy=image_to_field(Hmat,cx,cy)
                if xy: rec["x_m"], rec["y_m"] = xy
            ball_series.append(rec)

        out_idx+=1
        if out_idx>=max_frames: break

    cap.release()

    # TODO: smoothing/velocity/metrics/events layers here

    meta = {
        "fps": fps, "width": W, "height": H,
        "frame_skip": frame_skip, "homography": (Hmat.tolist() if Hmat is not None else None),
        "pitch_m": [105,68]
    }
    return {"meta": meta, "tracks": tracks, "ball": ball_series}
