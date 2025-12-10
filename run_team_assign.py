import argparse, json, numpy as np
from team_assign import TeamAssigner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    print(f"Loading {args.inp} ...")
    with open(args.inp) as f:
        data = json.load(f)

    frames = data.get("frames", [])
    print("Frames:", len(frames))
    if not frames:
        raise RuntimeError("No frames[] found")

    assigner = TeamAssigner()
    processed = 0

    for fr in frames:
        items = fr.get("items", [])
        if not items:
            continue
        assigner._frame_count += 1
        ids = [it.get("id") for it in items if "id" in it]
        bboxes = [it.get("bbox") for it in items if "bbox" in it]
        if not ids or not bboxes:
            continue
        # cluster every N frames
        if assigner._frame_count % assigner.cluster_every == 0:
            assigner._fit(np.zeros((720,1280,3), np.uint8), ids, bboxes)
        for it in items:
            pid = it.get("id")
            if pid is None: 
                continue
            team = assigner.get_team(pid)
            if team:
                it["team"] = team
        processed += 1
        if processed % 1000 == 0:
            print("Processed", processed, "frames...")

    print(f"Processed {processed} frames. Writing {args.out} ...")
    with open(args.out,"w") as f:
        json.dump({"frames": frames}, f)

if __name__ == "__main__":
    main()
