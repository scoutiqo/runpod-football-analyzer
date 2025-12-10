import csv, json, math
from collections import defaultdict

INP = "runs/json/frame_dataset_v1.csv"
OUT = "runs/json/model_v0.json"

LR = 0.05          # learning rate
L2 = 1e-4          # small weight decay
INIT = 0.0         # initial weight

w = defaultdict(lambda: INIT)
n = defaultdict(int)

with open(INP, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        hid = int(float(row["holder_id"]))
        rwd = float(row["reward"])
        # online update: pull weight toward observed reward
        w_i = w[hid]
        grad = (rwd - w_i) - L2*w_i
        w[hid] = w_i + LR * grad
        n[hid] += 1

# pack and export
model = {
    "type": "holder_reward_ewma",
    "lr": LR, "l2": L2, "init": INIT,
    "weights": {str(h): round(w[h], 6) for h in w},
    "counts": {str(h): n[h] for h in n}
}
open(OUT, "w").write(json.dumps(model, indent=2))

# quick report: top 10 holders by learned weight with >=50 samples
# build top10 BEFORE string-casting keys, using local n (int keys)
pairs = list(w.items())
pairs.sort(key=lambda kv: kv[1], reverse=True)
top10 = [ (round(val,6), n[k], k) for k,val in pairs if n[k] >= 50 ][:10]
print("Wrote", OUT)
print("top10:", top10)
