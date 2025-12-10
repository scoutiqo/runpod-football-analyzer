import argparse, json, os
import cv2
import numpy as np
from ultralytics import YOLO
from tracker_players import PlayerTracker

# COCO class id for sports ball
SPORTS_BALL_CLS = 32

def run(video_path, model_path, out_path, conf=0.30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    model = YOLO(model_path)  # e.g. yolov8n.pt
    tracker = PlayerTracker(conf_thresh=conf, person_ids=(0,))  # COCO person id = 0

    tracks = []
    fno = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fno / fps

        # --- YOLO inference
        res = model.predict(source=frame, verbose=False, conf=conf, iou=0.45, imgsz=960)[0]
        xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0,4), np.float32)
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.empty((0,), np.float32)
        clses = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.empty((0,), np.int32)

        # --- TRACK PLAYERS
        dets = {"xyxy": xyxy.tolist(), "conf": confs.tolist(), "cls": clses.tolist()}
        trk = tracker.update(dets)
        for p in trk:
            x1,y1,x2,y2 = p["x1"],p["y1"],p["x2"],p["y2"]
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            tracks.append({"t": round(t, 4), "type": "player", "id": int(p["id"]),
                           "x_px": float(cx), "y_px": float(cy)})

        # --- SIMPLE BALL PICK (highest-conf sports ball detection)
        if clses.size:
            mask_ball = (clses == SPORTS_BALL_CLS)
            if mask_ball.any():
                idx = np.argmax(confs[mask_ball])
                ball_box = xyxy[mask_ball][idx]
                bx1,by1,bx2,by2 = ball_box
                bc_x, bc_y = (bx1+bx2)/2.0, (by1+by2)/2.0
                tracks.append({"t": round(t, 4), "type": "ball",
                               "x_px": float(bc_x), "y_px": float(bc_y)})

        fno += 1

    cap.release()
    out = {"video_id": os.path.basename(video_path), "fps": fps, "tracks": tracks}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[make_tracks_raw] wrote {out_path}  items={len(tracks)}   fps={fps:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--out", default="tracks_raw.json")
    ap.add_argument("--conf", type=float, default=0.30)
    args = ap.parse_args()
    run(args.video, args.model, args.out, conf=args.conf)
