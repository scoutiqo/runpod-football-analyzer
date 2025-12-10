import csv, json, math
from collections import defaultdict, Counter

INP = "runs/json/frame_dataset_v1.csv"
OUT = "runs/json/policy_v0.json"

tot_r = defaultdict(float)
cnt_r = defaultdict(int)
tot_r_team = {0: defaultdict(float), 1: defaultdict(float)}
cnt_r_team = {0: defaultdict(int), 1: defaultdict(int)}
seen_holders = Counter()

with open(INP, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        hid = int(float(row["holder_id"]))
        ht_raw = row["holder_team"]
        ht = None if ht_raw=="" or ht_raw=="None" else int(float(ht_raw))
        rw = float(row["reward"])
        tot_r[hid] += rw
        cnt_r[hid] += 1
        seen_holders[hid] += 1
        if ht in (0,1):
            tot_r_team[ht][hid] += rw
            cnt_r_team[ht][hid] += 1

def topk(dtot, dcnt, k=25, min_n=30):
    items=[]
    for hid, s in dtot.items():
        n = dcnt.get(hid, 0)
        if n >= min_n:
            items.append((s/n, n, hid))
    items.sort(reverse=True)  # by mean reward
    return [{"holder_id": hid, "mean_reward": round(m,4), "samples": n} for m,n,hid in items[:k]]

policy = {
    "global_top": topk(tot_r, cnt_r, k=50, min_n=50),
    "team0_top":  topk(tot_r_team[0], cnt_r_team[0], k=30, min_n=30),
    "team1_top":  topk(tot_r_team[1], cnt_r_team[1], k=30, min_n=30),
    "stats": {
        "unique_holders": len(seen_holders),
        "holders_ge_50_samples": sum(1 for h in seen_holders if seen_holders[h] >= 50)
    }
}

open(OUT, "w").write(json.dumps(policy, indent=2))
print("Wrote", OUT)
print(json.dumps(policy["stats"], indent=2))
print("global_top_5:", policy["global_top"][:5])
