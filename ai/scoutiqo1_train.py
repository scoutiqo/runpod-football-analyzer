import argparse, csv, json, os
import math
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/json/frame_dataset_v1.csv")
    ap.add_argument("--out", default="runs/json/dataset_stats.json")
    args = ap.parse_args()

    n=0
    holder_ids=Counter()
    holder_teams=Counter()
    sum_r=sum_r0=sum_r1=0.0
    min_t=10**9; max_t=-1

    # quick bounds for ball coords to understand scale
    min_bx=min_by=10**9
    max_bx=max_by=-10**9

    with open(args.data, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            n+=1
            t=int(float(row["t"]))
            bx=float(row["ball_x"]); by=float(row["ball_y"])
            hid=int(float(row["holder_id"])); 
            ht=row["holder_team"]
            ht = None if ht=="" or ht=="None" else int(float(ht))
            rwd=float(row["reward"]); r0=float(row["team0_r"]); r1=float(row["team1_r"])

            holder_ids[hid]+=1
            holder_teams[ht]+=1
            sum_r+=rwd; sum_r0+=r0; sum_r1+=r1
            min_t=min(min_t,t); max_t=max(max_t,t)
            min_bx=min(min_bx,bx); max_bx=max(max_bx,bx)
            min_by=min(min_by,by); max_by=max(max_by,by)

    stats={
        "rows": n,
        "t_range": [min_t, max_t],
        "ball_x_range": [min_bx, max_bx],
        "ball_y_range": [min_by, max_by],
        "unique_holder_ids": len(holder_ids),
        "holder_id_top10": holder_ids.most_common(10),
        "holder_team_counts": {str(k): v for k, v in holder_teams.items()},
        "reward_sum": round(sum_r,2),
        "team0_reward_sum": round(sum_r0,2),
        "team1_reward_sum": round(sum_r1,2),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w") as f:
        json.dump(stats,f,indent=2)
    print("Wrote", args.out)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
