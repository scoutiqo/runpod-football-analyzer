import json, argparse

def to_flat(inp, outp):
    with open(inp, "r") as f:
        data = json.load(f)

    # allow either list of frames or {"frames":[...]}
    frames = data.get("frames", data if isinstance(data, list) else [])
    flat = []
    for fr in frames:
        t = fr.get("frame")
        items = fr.get("items", [])
        for it in items:
            x1, y1, x2, y2 = it["bbox_xyxy"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = (x2 - x1)
            h  = (y2 - y1)
            cls = "person" if it.get("cls",0)==0 else ("ball" if it.get("cls")==1 else str(it.get("cls")))
            flat.append({
                "t": int(t),
                "id": int(it.get("id", -1)),
                "cls": cls,
                "x": float(cx),
                "y": float(cy),
                "w": float(w),
                "h": float(h),
                "conf": float(it.get("conf", 0.0))
            })
    with open(outp, "w") as f:
        json.dump({"tracks": flat}, f)
    print(f"Wrote {outp} with {len(flat)} rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"), ap.add_argument("outp")
    a = ap.parse_args()
    to_flat(a.inp, a.outp)
