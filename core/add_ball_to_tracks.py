import argparse, json
from pathlib import Path

import cv2
from ultralytics import YOLO

def add_ball(video_path, tracks_path, out_path, model_path, ball_class, conf):
    tracks_path = Path(tracks_path)
    out_path = Path(out_path)
    data = json.loads(tracks_path.read_text())
    frames = data.get("frames") or data.get("by_frame") or data.get("data") or []
    if isinstance(frames, dict):
        # If by_frame is a dict with frame indices as keys
        try:
            frames = [frames[k] for k in sorted(frames.keys(), key=lambda x: int(x))]
        except Exception:
            frames = list(frames.values())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    model = YOLO(model_path)

    idx = 0
    frames_with_ball = 0
    while idx < len(frames):
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, imgsz=640, conf=conf, verbose=False)[0]
        best = None
        if res.boxes is not None and len(res.boxes) > 0:
            b = res.boxes
            import numpy as np
            xyxy = b.xyxy.cpu().numpy()
            scores = b.conf.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)

            mask = cls == ball_class
            if mask.any():
                idxs = np.where(mask)[0]
                best_i = int(idxs[scores[mask].argmax()])
                x1, y1, x2, y2 = xyxy[best_i].tolist()
                score = float(scores[best_i])
                best = {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cls": int(ball_class), "conf": score
                }

        fr = frames[idx]
        fr["ball"] = best
        if best is not None:
            frames_with_ball += 1

        idx += 1

    cap.release()

    data["frames"] = frames
    out_path.write_text(json.dumps(data))
    print(f"Wrote {out_path} with {len(frames)} frames, ball in {frames_with_ball} frames")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video (e.g. uploads/test_short.mp4)")
    ap.add_argument("--tracks", required=True, help="Path to wrapped tracks JSON")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO model path (one that sees sports ball)")
    ap.add_argument("--ball_class", type=int, default=32, help="Class id for sports ball")
    ap.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    args = ap.parse_args()

    add_ball(
        video_path=args.video,
        tracks_path=args.tracks,
        out_path=args.out,
        model_path=args.model,
        ball_class=args.ball_class,
        conf=args.conf,
    )

if __name__ == "__main__":
    main()
