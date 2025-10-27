# /training/train.py
import os, sys, json, time, shutil
from pathlib import Path
from typing import List, Optional
from ultralytics import YOLO

from training.generate_pseudolabels import build_from_videos
from training.utils_supabase import upload_folder, DEFAULT_BUCKET

def jprint(obj): print(json.dumps(obj, ensure_ascii=False), flush=True)

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def train_main(
    dataset: str,
    model: str = "yolov8n.pt",
    epochs: int = 10,
    batch: int = 8,
    imgsz: int = 640,
    project: str = "/tmp/train",
    name: str = "train_run",
    resume: bool = False,
    device: str = "cpu",
    progress_cb=None
):
    out_dir = Path(project)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    ckpt_dir = out_dir / "checkpoints"
    logs_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    jsonl = logs_dir / "train.jsonl"

    # Load YOLO model
    yolo_model = YOLO(model)

    # Attach callbacks for epoch logging
    epoch_state = {"last_ckpt": None, "best_ckpt": None}

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
        
        # Call progress callback if provided
        if progress_cb:
            progress_cb({
                "epoch": epoch,
                "epochs": max_epoch,
                "train_loss": metrics['train_loss'],
                "lr": metrics['lr']
            })

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
            epoch_state["best_ckpt"] = str(dst)
            print(f"CHECKPOINT:{dst.as_posix()}", flush=True)

    yolo_model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    yolo_model.add_callback("on_model_save", on_model_save)

    # Real training (no mocks)
    results = yolo_model.train(
        data=dataset,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(out_dir),
        name=name,
        pretrained=True,
        verbose=True,
        device=device,
        resume=resume
    )

    # Find results directory
    results_dir = None
    for item in out_dir.glob(f"**/{name}"):
        if item.is_dir():
            results_dir = str(item)
            break
    
    # Find curves PNG
    curves_png = None
    if results_dir:
        for png_file in Path(results_dir).glob("**/*.png"):
            if "results" in png_file.name.lower():
                curves_png = str(png_file)
                break

    # Return artifacts dict
    artifacts = {
        "best_ckpt": epoch_state.get("best_ckpt"),
        "last_ckpt": epoch_state.get("last_ckpt"),
        "curves_png": curves_png,
        "results_dir": results_dir
    }

    print(f"RESULT_DIR:{out_dir.as_posix()}", flush=True)
    return artifacts

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
