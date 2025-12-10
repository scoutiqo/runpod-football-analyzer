import csv, json, argparse, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PassDS(Dataset):
    def __init__(self, path, seed=42, val_frac=0.1):
        rows=[]
        with open(path, newline="") as f:
            r=csv.DictReader(f)
            for row in r:
                try:
                    bx=float(row["ball_x"]); by=float(row["ball_y"])
                    fid=int(float(row["from_id"]))
                    tid=int(float(row["to_id"]))
                    y =int(float(row["success"]))
                    rows.append((bx,by,fid,tid,y))
                except: pass
        # split
        import random
        rnd=random.Random(seed); rnd.shuffle(rows)
        n=len(rows); n_val=max(1,int(n*val_frac))
        self.val = rows[:n_val]; self.train = rows[n_val:]
        # holder dictionary across both splits
        hids=sorted({r[2] for r in rows} | {r[3] for r in rows})
        self.h2i={h:i for i,h in enumerate(hids)}
    def loaders(self, bs=4096):
        def to_dl(x):
            X=[(bx,by,self.h2i[f],self.h2i[t],y) for (bx,by,f,t,y) in x]
            class _D(Dataset):
                def __init__(self, X): self.X=X
                def __len__(self): return len(self.X)
                def __getitem__(self,i):
                    bx,by,f,t,y=self.X[i]
                    return (torch.tensor([bx,by],dtype=torch.float32),
                            torch.tensor(f,dtype=torch.long),
                            torch.tensor(t,dtype=torch.long),
                            torch.tensor([y],dtype=torch.float32))
            return DataLoader(_D(X), batch_size=bs, shuffle=True, drop_last=False)
        return to_dl(self.train), to_dl(self.val), len(self.h2i)

class PassModel(nn.Module):
    def __init__(self, n_h, emb=16):
        super().__init__()
        self.emb = nn.Embedding(n_h, emb)
        self.net = nn.Sequential(
            nn.Linear(emb*2+2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, ball, f_idx, t_idx):
        ef = self.emb(f_idx); et = self.emb(t_idx)
        x = torch.cat([ball, ef, et], dim=1)
        return self.net(x)

def train(csv_path, epochs=8, emb=16, lr=1e-3, bs=4096, out="runs/json/pass_model_v1.pt", report="runs/json/pass_train_v1.json"):
    ds=PassDS(csv_path)
    tr, va, n_h = ds.loaders(bs=bs)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=PassModel(n_h, emb=emb).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce=nn.BCELoss()
    hist=[]
    def eval_auc(dl):
        import math
        ys=[]; ps=[]
        with torch.no_grad():
            for ball,f,t,y in dl:
                y=y.to(device); p=model(ball.to(device), f.to(device), t.to(device))
                ys += y.view(-1).tolist(); ps += p.view(-1).tolist()
        # AUC (simple Mann–Whitney U)
        pairs=0; u=0
        pos=[p for p,y in zip(ps,ys) if y>0.5]
        neg=[p for p,y in zip(ps,ys) if y<=0.5]
        if not pos or not neg: return None
        pos.sort(); neg.sort()
        import bisect
        for p in pos:
            u += bisect.bisect(neg, p-1e-12)
            pairs += len(neg)
        return u/max(1,pairs)
    for ep in range(1,epochs+1):
        model.train(); tot=0; n=0
        for ball,f,t,y in tr:
            ball,f,t,y = ball.to(device), f.to(device), t.to(device), y.to(device)
            p = model(ball,f,t)
            loss=bce(p,y); opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*y.size(0); n+=y.size(0)
        va_auc=eval_auc(va)
        hist.append({"epoch":ep, "train_bce":tot/max(1,n), "val_auc":va_auc})
        print(f"[ep {ep}] train_bce={tot/max(1,n):.4f}  val_auc={va_auc if va_auc is not None else 'NA'}")
    torch.save({"state_dict":model.state_dict(),"n_holders":n_h,"emb":emb}, out)
    json.dump({"history":hist,"n_holders":n_h,"emb":emb}, open(report,"w"), indent=2)
    print("Saved", out, "and", report)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/json/pass_success_v1.csv")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--out", default="runs/json/pass_model_v1.pt")
    ap.add_argument("--report", default="runs/json/pass_train_v1.json")
    args=ap.parse_args()
    train(args.csv, args.epochs, args.emb, args.lr, args.bs, args.out, args.report)
