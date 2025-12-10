import json, argparse, math, sys

def to_center(xyxy):
    x1,y1,x2,y2 = xyxy
    w = x2 - x1; h = y2 - y1
    return x1 + w/2, y1 + h/2, w, h

def from_frames_struct(data, player_cls=0, ball_cls=1):
    out=[]
    for f in data.get("frames", []):
        t = f.get("frame")
        for it in f.get("items", []):
            # Some dumps use ints/floats, keep float for centers
            x,y,w,h = to_center(it["bbox_xyxy"])
            out.append({
                "t": int(t),
                "id": int(it["id"]),
                "cls": int(it["cls"]),
                "x": float(x), "y": float(y),
                "w": float(w), "h": float(h),
                "conf": float(it.get("conf", 1.0)),
            })
    return out

def from_flat_struct(data):
    # Already flat — just ensure keys exist and types are OK
    out=[]
    for r in data:
        out.append({
            "t": int(r["t"]),
            "id": int(r["id"]),
            "cls": int(r["cls"]),
            "x": float(r["x"]), "y": float(r["y"]),
            "w": float(r.get("w", 0.0)), "h": float(r.get("h", 0.0)),
            "conf": float(r.get("conf", 1.0)),
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--player-cls", type=int, default=0)
    ap.add_argument("--ball-cls", type=int, default=1)
    args = ap.parse_args()

    with open(args.inp, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "frames" in data:
        flat = from_frames_struct(data, args.player_cls, args.ball_cls)
    elif isinstance(data, list):
        flat = from_flat_struct(data)
    else:
        print("Unrecognized schema. Expect dict with 'frames' or flat list.", file=sys.stderr)
        sys.exit(2)

    flat.sort(key=lambda r:(r["t"], r["id"]))
    with open(args.out, "w") as f:
        json.dump(flat, f)
    print(f"Wrote {args.out} with {len(flat)} rows")

if __name__ == "__main__":
    main()
