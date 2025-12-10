import json, os, math, cv2, argparse
from pathlib import Path

# class map (start simple; you'll correct in labeling)
# 0=player (prefill), 1=referee, 2=goalkeeper, 3=ball (you'll add in the tool)
CLS_MAP = {"player":0}

def yolo_line(x1,y1,x2,y2,w,h,cls=0):
    # clamp
    x1=max(0,x1); y1=max(0,y1); x2=min(w-1,x2); y2=min(h-1,y2)
    bw = max(1, x2-x1); bh = max(1, y2-y1)
    cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
    return f"{cls} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_id", required=True, help="Job ID folder under raw-videos")
    ap.add_argument("--video", required=False, help="Path to the source video (used to grab frames)")
    ap.add_argument("--tracks", required=False, help="Path to tracks.json (detections)")
    ap.add_argument("--every_n", type=int, default=5, help="sample every Nth frame")
    ap.add_argument("--out", default="datasets/scoutiqo_seed", help="output root")
    args = ap.parse_args()

    out_root = Path(args.out)
    img_dir  = out_root/"images"/"train"
    lbl_dir  = out_root/"labels"/"train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # defaults from Supabase paths if not provided
    if not args.video:
        args.video  = f"runs/videos/out_ids.mp4"  # fall back to local render (same size)
    if not args.tracks:
        args.tracks = f"runs/json/tracks.json"

    # load detections (expects list of {t, x1,y1,x2,y2, class?})
    with open(args.tracks, "r") as f:
        tracks = json.load(f)

    # group by frame index t
    by_t = {}
    for r in tracks:
        t = r.get("t")
        if t is None: continue
        by_t.setdefault(t, []).append(r)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    kept = 0
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        if t % args.every_n == 0:
            # save image
            img_path = img_dir / f"frame_{t:06d}.jpg"
            cv2.imwrite(str(img_path), frame)

            # write labels (prefill: everyone is 'player' = 0)
            lbl_path = lbl_dir / f"frame_{t:06d}.txt"
            with open(lbl_path, "w") as lf:
                for r in by_t.get(t, []):
                    x1,y1,x2,y2 = r["x1"], r["y1"], r["x2"], r["y2"]
                    lf.write(yolo_line(x1,y1,x2,y2,w,h,cls=0) + "\n")
            kept += 1
        t += 1

    cap.release()

    # write data.yaml
    (out_root/"data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/train\n"
        "names:\n"
        "  0: player\n"
        "  1: referee\n"
        "  2: goalkeeper\n"
        "  3: ball\n"
    )
    print(f"Exported {kept} images with prefilled labels to {out_root}")
    print("Next: open Label Studio, correct classes, add missing players, refs and ball.")
if __name__ == "__main__":
    main()
