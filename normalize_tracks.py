import json, sys

inp = sys.argv[1]
out = sys.argv[2]

with open(inp, "r", encoding="utf-8") as f:
    tj = json.load(f)

# Accept either:
#  - list of track records
#  - {"tracks":[...], ...}
if isinstance(tj, list):
    tracks = tj
elif isinstance(tj, dict) and "tracks" in tj:
    tracks = tj["tracks"]
else:
    # best effort fallback
    tracks = tj.get("data", tj.get("results", tj))

# Compute duration if possible from timestamps
def safe_times(tracks):
    ts = [float(r.get("t", 0.0)) for r in tracks if isinstance(r, dict) and "t" in r]
    if not ts: return 0.0
    return max(ts) - min(ts)

duration_s = safe_times(tracks)

norm = {
    "video": {"duration_s": duration_s},
    "tracks": tracks
}

with open(out, "w", encoding="utf-8") as f:
    json.dump(norm, f, ensure_ascii=False)
print(f"wrote {out} with {len(tracks)} tracks, duration_s≈{duration_s:.2f}")
