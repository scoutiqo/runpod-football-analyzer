import json, sys

inp = sys.argv[1]
out = sys.argv[2]

with open(inp, "r", encoding="utf-8") as f:
    tj = json.load(f)

# Accept either raw list, or dict with "tracks"
if isinstance(tj, list):
    tracks = tj
elif isinstance(tj, dict) and "tracks" in tj:
    tracks = tj["tracks"]
else:
    tracks = tj.get("data", tj.get("results", tj))

# Ensure fields are present
normed = []
for r in tracks:
    if not isinstance(r, dict):
        continue
    rec = dict(r)
    if "type" not in rec:
        rec["type"] = "player" if "id" in rec else "ball"
    if "id" not in rec and rec["type"] == "player":
        rec["id"] = -1
    if "t" not in rec:
        rec["t"] = 0.0
    normed.append(rec)

# Sort by time
normed = sorted(normed, key=lambda z: z.get("t", 0.0))

# Estimate duration
def safe_times(arr):
    ts = [float(r.get("t", 0.0)) for r in arr if "t" in r]
    if not ts:
        return 0.0
    return max(ts) - min(ts)

duration_s = safe_times(normed)

outj = {
    "video": {"duration_s": duration_s},
    "tracks": normed
}

with open(out, "w", encoding="utf-8") as f:
    json.dump(outj, f, ensure_ascii=False)

print(f"wrote {out} with {len(normed)} tracks, duration≈{duration_s:.2f}s")
