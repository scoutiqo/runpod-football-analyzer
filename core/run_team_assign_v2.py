import argparse, json, cv2, numpy as np
from team_assign import TeamAssigner

def get_id(it):
    for k in ("id","track_id","tid"):
        if k in it: return it[k]
    return None

def get_bbox(it):
    if "bbox" in it and isinstance(it["bbox"], (list,tuple)) and len(it["bbox"])==4:
        x1,y1,x2,y2 = it["bbox"]
        return [int(x1),int(y1),int(x2),int(y2)]
    # common alternative layouts
    keys = it.keys()
    if all(k in it for k in ("x1","y1","x2","y2")):
        return [int(it["x1"]),int(it["y1"]),int(it["x2"]),int(it["y2"])]
    if "xyxy" in it and len(it["xyxy"])==4:
        x1,y1,x2,y2 = it["xyxy"]
        return [int(x1),int(y1),int(x2),int(y2)]
    if "box" in it and len(it["box"])==4:
        x1,y1,w,h = it["box"]
        return [int(x1),int(y1),int(x1+w),int(y1+h)]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--every", type=int, default=15, help="cluster every N frames")
    args = ap.parse_args()

    data = json.load(open(args.inp))
    frames = data.get("frames", [])
    video_path = data.get("video")
    if not frames: raise RuntimeError("No frames[] found")
    if not video_path: raise RuntimeError("No video path in JSON")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    assigner = TeamAssigner(cluster_every=args.every)
    processed=0; fit_count=0

    for fr in frames:
        fidx = fr.get("frame", None)
        items = fr.get("items", [])
        if fidx is None or not items:
            continue

        # collect ids+bboxes for this frame
        ids=[]; bbs=[]
        for it in items:
            pid = get_id(it)
            bb = get_bbox(it)
            if pid is not None and bb is not None:
                ids.append(pid); bbs.append(bb)

        if not ids:
            continue

        assigner._frame_count += 1

        # When clustering, read the actual frame from the video for color extraction
        if assigner._frame_count % assigner.cluster_every == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame_img = cap.read()
            if not ok:
                # fallback to a black frame but still proceed
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                frame_img = np.zeros((max(h,1), max(w,1), 3), np.uint8)
            assigner._fit(frame_img, ids, bbs)
            fit_count += 1

        # assign labels for this frame
        for it in items:
            pid = get_id(it)
            if pid is None: 
                continue
            team = assigner.get_team(pid)
            if team:
                it["team"] = str(team)

        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed} frames... (fits={fit_count})")

    cap.release()
    print(f"Processed {processed} frames total. KMeans fits: {fit_count}. Writing {args.out} ...")
    json.dump({"video": video_path, "frames": frames}, open(args.out,"w"))
