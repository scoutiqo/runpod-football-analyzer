import json, sys

def load_tracks(p):
    with open(p, "r", encoding="utf-8") as f:
        tj = json.load(f)
    if isinstance(tj, list):
        return tj
    if isinstance(tj, dict) and "tracks" in tj:
        return tj["tracks"]
    return tj.get("data", tj.get("results", tj))

def float_or_none(v):
    try:
        return float(v)
    except:
        return None

def infer_type(r: dict) -> str:
    t = r.get("type")
    if t in ("player","ball"):
        return t
    cls = (r.get("cls") or r.get("label") or r.get("class") or "").lower()
    if r.get("is_ball") or "ball" in cls:
        return "ball"
    if cls in ("person","player","human","person_yolo","person_detect"):
        return "player"
    for k in ("id","track_id","tid","pid","obj_id","track","trk_id"):
        if k in r:
            return "player"
    return "player"

def infer_id(r: dict) -> int:
    for k in ("id","track_id","tid","pid","obj_id","track","trk_id"):
        if k in r:
            try:
                return int(r[k])
            except:
                return -1
    return -1

def pick_coords(r: dict):
    x_m = float_or_none(r.get("x_m")); y_m = float_or_none(r.get("y_m"))
    if x_m is None and "x" in r and "y" in r:
        xm = float_or_none(r.get("x")); ym = float_or_none(r.get("y"))
        if xm is not None and ym is not None:
            return ("m", xm, ym)
    if x_m is not None and y_m is not None:
        return ("m", x_m, y_m)
    x_px = float_or_none(r.get("x_px")); y_px = float_or_none(r.get("y_px"))
    if x_px is not None and y_px is not None:
        return ("px", x_px, y_px)
    cx = float_or_none(r.get("cx")); cy = float_or_none(r.get("cy"))
    if cx is not None and cy is not None:
        return ("px", cx, cy)
    return (None, None, None)

def main(inp, outp):
    raw = load_tracks(inp)
    norm = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        t = float_or_none(r.get("t"))
        if t is None:
            continue
        coord_kind, x, y = pick_coords(r)
        if coord_kind is None:
            continue
        item = {"t": t}
        ty = infer_type(r)
        item["type"] = ty
        if ty == "player":
            item["id"] = infer_id(r)
        if coord_kind == "m":
            item["x_m"], item["y_m"] = x, y
        else:
            item["x_px"], item["y_px"] = x, y
        norm.append(item)
    norm.sort(key=lambda z: z["t"])
    duration = (norm[-1]["t"] - norm[0]["t"]) if norm else 0.0
    outj = {"video": {"duration_s": duration}, "tracks": norm}
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(outj, f, ensure_ascii=False)
    print(f"Normalized {len(raw)} -> {len(norm)} tracks, duration≈{duration:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python normalize_tracks_v3.py <input.json> <output.json>")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
