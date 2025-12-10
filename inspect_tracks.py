# inspect_tracks.py
import json, collections, sys

with open("tracks.json","r",encoding="utf-8") as f:
    tj = json.load(f)

if isinstance(tj, list):
    tracks = tj
else:
    tracks = tj.get("tracks", [])

print("total:", len(tracks))
ctr_types = collections.Counter()
ctr_keys  = collections.Counter()
ids = 0

for r in tracks[:1000]:
    if isinstance(r, dict):
        ctr_types[r.get("type","<missing>")] += 1
        ctr_keys.update(r.keys())
        if "id" in r:
            ids += 1

print("types:", ctr_types)
print("has id count:", ids)
print("top keys:", [k for k,_ in ctr_keys.most_common(20)])
print("sample:", tracks[0] if tracks else {})