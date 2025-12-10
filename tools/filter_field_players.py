import json
import argparse

def inside_pitch(x_m, y_m):
    """Return True if coordinates are inside a real pitch."""
    if x_m is None or y_m is None:
        return False
    return 0 <= x_m <= 68 and 0 <= y_m <= 105

def run(inp, outp):
    with open(inp, "r") as f:
        data = json.load(f)

    # Determine format
    if isinstance(data, dict) and "tracks" in data:
        tracks = data["tracks"]
    else:
        tracks = data

    total = len(tracks)
    cleaned = []

    for row in tracks:
        if row.get("type") != "player":
            continue

        x_m = row.get("x_m", None)
        y_m = row.get("y_m", None)

        # reject off-pitch
        if not inside_pitch(x_m, y_m):
            continue

        # reject tiny detections (junk)
        h = row.get("h_px", None)
        if h is not None and h < 60:
            continue

        # reject crazy speeds
        s = row.get("speed_ms", None)
        if isinstance(s, (float, int)) and s > 12:
            continue

        cleaned.append(row)

    # Write output matching original format
    if isinstance(data, dict) and "tracks" in data:
        data["tracks"] = cleaned
        with open(outp, "w") as f:
            json.dump(data, f, indent=2)
    else:
        with open(outp, "w") as f:
            json.dump(cleaned, f, indent=2)

    print(f"[filter] Total rows: {total}")
    print(f"[filter] Kept rows:  {len(cleaned)}")
    print(f"[filter] Removed:     {total - len(cleaned)}")
    print(f"[filter] Cleaned file → {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()
    run(args.inp, args.outp)
