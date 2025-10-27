# /training/train.py
import os, sys, json, time, shutil
from pathlib import Path
from typing import List, Optional
from ultralytics import YOLO

from training.generate_pseudolabels import build_from_videos
from training.utils_supabase import upload_folder, DEFAULT_BUCKET

def jprint(obj): print(json.dumps(obj, ensure_ascii=False), flush=True)

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def train(
    model_base: str,
    out_dir: Path,
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 8,
    lr0: float = 1e-3,
    dataset_yaml: Optional[Path] = None,
    video_paths: Optional[List[str]] = None,
    pseudolabel_conf: float = 0.4,
    pseudolabel_every_n: int = 5
):
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    ckpt_dir = out_dir / "checkpoints"
    logs_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    jsonl = logs_dir / "train.jsonl"

    # Prepare dataset
    if dataset_yaml is None:
        if not video_paths:
            raise SystemExit("Either dataset_yaml or video_paths must be provided.")
        ds_root = out_dir / "pseudolabel_dataset"
        ds_yaml = build_from_videos(video_paths, ds_root, model_base, conf=pseudolabel_conf, every_n=pseudolabel_every_n)
        dataset_yaml = ds_yaml

    model = YOLO(model_base)

    # Attach callbacks for epoch logging
    epoch_state = {"last_ckpt": None}

    def on_fit_epoch_end(trainer):
        # Ultralytics passes a trainer with stats; extract real values
        epoch = trainer.epoch + 1
        max_epoch = trainer.epochs
        metrics = {
            "t": now(),
            "epoch": epoch,
            "epochs": max_epoch,
            "train_loss": float(trainer.loss_items[0]) if trainer.loss_items else None,
            "lr": float(trainer.lr[0]) if isinstance(trainer.lr, (list, tuple)) else float(trainer.lr),
        }
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        print(f"TRAIN_EPOCH:{epoch}/{max_epoch} loss={metrics['train_loss']} lr={metrics['lr']}", flush=True)

    def on_model_save(trainer):
        # Copy last.pt/best.pt into our checkpoints folder and print a machine-readable line
        src_last = Path(trainer.save_dir) / "weights" / "last.pt"
        src_best = Path(trainer.save_dir) / "weights" / "best.pt"
        if src_last.exists():
            dst = ckpt_dir / f"epoch_{trainer.epoch+1:02d}_last.pt"
            shutil.copy2(src_last, dst)
            epoch_state["last_ckpt"] = str(dst)
            print(f"CHECKPOINT:{dst.as_posix()}", flush=True)
        if src_best.exists():
            dst = ckpt_dir / "best.pt"
            shutil.copy2(src_best, dst)
            print(f"CHECKPOINT:{dst.as_posix()}", flush=True)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_model_save", on_model_save)

    # Real training (no mocks)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        project=str(out_dir),
        name="ultra_run",
        pretrained=True,
        verbose=True
    )

    # Print a final pointer the handler/server can use
    print(f"RESULT_DIR:{out_dir.as_posix()}", flush=True)
    return out_dir

if __name__ == "__main__":
    # Minimal CLI for RunPod handler; parse env/argv lightly
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--dataset_yaml", default=None)
    ap.add_argument("--videos_json", default=None, help="JSON list of local video paths")
    ap.add_argument("--pseudolabel_conf", type=float, default=0.4)
    ap.add_argument("--pseudolabel_every_n", type=int, default=5)
    args = ap.parse_args()

    vids = None
    if args.videos_json:
        vids = json.loads(args.videos_json)

    out = train(
        model_base=args.model_base,
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        dataset_yaml=Path(args.dataset_yaml) if args.dataset_yaml else None,
        video_paths=vids,
        pseudolabel_conf=args.pseudolabel_conf,
        pseudolabel_every_n=args.pseudolabel_every_n
    )
    jprint({"ok": True, "out_dir": str(out)})
