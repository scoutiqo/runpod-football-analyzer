
import os, csv, json, math, random, argparse
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Data utils ----------
def load_rows(path):
    rows=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                t  = int(float(row["t"]))
                bx = float(row["ball_x"]); by=float(row["ball_y"])
                hid= int(float(row["holder_id"]))
                rw = float(row["reward"])
                rows.append((t,bx,by,hid,rw))
            except Exception:
                continue
    return rows

def split_and_index(rows, holders_min_samples=20, seed=42, val_frac=0.1):
    # filter holders with enough samples
    cnt={}
    for *_,hid,_ in rows: cnt[hid]=cnt.get(hid,0)+1
    rows=[r for r in rows if cnt[r[3]]>=holders_min_samples]
    # split
    rnd=random.Random(seed); rnd.shuffle(rows)
    n=len(rows); n_val=max(1,int(n*val_frac))
    val_rows   = rows[:n_val]
    train_rows = rows[n_val:]
    # unified holder-id mapping from BOTH splits
    all_hids = sorted({r[3] for r in train_rows} | {r[3] for r in val_rows})
    h2i = {h:i for i,h in enumerate(all_hids)}
    # map to indices
    train_rows_idx = [(t,bx,by,h2i[hid],rw) for (t,bx,by,hid,rw) in train_rows]
    val_rows_idx   = [(t,bx,by,h2i[hid],rw) for (t,bx,by,hid,rw) in val_rows]
    return train_rows_idx, val_rows_idx, len(h2i)

class FrameDS(Dataset):
    def __init__(self, rows_idx):
        # rows_idx are (t,bx,by,hid_idx,rw)
        self.rows = rows_idx
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        t,bx,by,hid_idx,rw = self.rows[idx]
        return (
            torch.tensor([bx,by], dtype=torch.float32),
            torch.tensor(hid_idx,   dtype=torch.long),
            torch.tensor([rw],      dtype=torch.float32),
        )

def build_loaders(path, holders_min_samples=20, seed=42, val_frac=0.1, bs=1024):
    rows = load_rows(path)
    train_rows, val_rows, n_h = split_and_index(rows, holders_min_samples, seed, val_frac)
    dl_train = DataLoader(FrameDS(train_rows), batch_size=bs, shuffle=True,  drop_last=False)
    dl_val   = DataLoader(FrameDS(val_rows),   batch_size=bs, shuffle=False, drop_last=False)
    return dl_train, dl_val, n_h

# ---------- Model ----------
class SupModel(nn.Module):
    def __init__(self, n_holders, emb_dim=16):
        super().__init__()
        self.emb = nn.Embedding(n_holders, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim+2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, ball, hid_idx):
        e = self.emb(hid_idx)           # [B, D]
        x = torch.cat([ball, e], dim=1) # [B, 2+D]
        return self.mlp(x)

# ---------- Train/Eval ----------
def train_one_epoch(model, opt, crit, dl, device):
    model.train(); tot=0.0; n=0
    for ball, hid, y in dl:
        ball=ball.to(device); hid=hid.to(device); y=y.to(device)
        yhat=model(ball, hid)
        loss=crit(yhat, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*y.size(0); n += y.size(0)
    return tot/max(1,n)

@torch.no_grad()
def eval_epoch(model, crit, dl, device):
    model.eval(); tot=0.0; n=0
    for ball, hid, y in dl:
        ball=ball.to(device); hid=hid.to(device); y=y.to(device)
        yhat=model(ball, hid)
        loss=crit(yhat, y)
        tot += loss.item()*y.size(0); n += y.size(0)
    return tot/max(1,n)

# ---------- Main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/json/frame_dataset_v1.csv")
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--min_samples", type=int, default=20)
    ap.add_argument("--out", default="runs/json/model_sup_v1.pt")
    ap.add_argument("--report", default="runs/json/training_sup_v1.json")
    args=ap.parse_args()

    device="cuda" if torch.cuda.is_available() else "cpu"
    train, val, n_h = build_loaders(args.csv, holders_min_samples=args.min_samples, bs=args.bs)
    model=SupModel(n_holders=n_h, emb_dim=args.emb).to(device)
    crit=nn.MSELoss()
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    hist=[]
    for ep in range(1, args.epochs+1):
        tr=train_one_epoch(model,opt,crit,train,device)
        va=eval_epoch(model,crit,val,device)
        hist.append({"epoch":ep,"train_mse":tr,"val_mse":va})
        print(f"[ep {ep}] train_mse={tr:.6f}  val_mse={va:.6f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"n_holders":n_h,"emb_dim":args.emb}, args.out)
    with open(args.report,"w") as f:
        json.dump({"history":hist,"n_holders":n_h,"emb_dim":args.emb}, f, indent=2)
    print("Saved", args.out, "and", args.report)

if __name__=="__main__":
    main()
