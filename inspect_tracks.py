<<<<<<< HEAD
# inspect_tracks.py
import json, collections, sys
=======
﻿import json, collections
>>>>>>> 7e67776 (chore: apply auto-tuned params from ai_agent)

with open("tracks.json","r",encoding="utf-8") as f:
    tj = json.load(f)

tracks = tj.get("tracks", tj if isinstance(tj, list) else [])

print("total:", len(tracks))
ctr_types = collections.Counter()
ctr_keys  = collections.Counter()
ids = 0

for r in tracks[:1000]:
    if isinstance(r, dict):
        ctr_types[r.get("type","<missing>")] += 1
        ctr_keys.update(r.keys())
<<<<<<< HEAD
        if "id" in r: 
=======
        if "id" in r:
>>>>>>> 7e67776 (chore: apply auto-tuned params from ai_agent)
            ids += 1

print("types:", ctr_types)
print("has id count:", ids)
<<<<<<< HEAD
print("top keys:", [k for k,_ in ctr_keys.most_common(15)])
=======
print("top keys:", [k for k,_ in ctr_keys.most_common(20)])
>>>>>>> 7e67776 (chore: apply auto-tuned params from ai_agent)
print("sample:", tracks[0] if tracks else {})
