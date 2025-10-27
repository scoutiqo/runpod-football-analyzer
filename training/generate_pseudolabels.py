# /training/generate_pseudolabels.py
import os, cv2, json, math
from pathlib import Path
from typing import List, Tuple
from ultralytics import YOLO

# YOLO class mapping: set the classes you care about (e.g., player=0, ball=1)
# Adjust to your current model's class indices
CLASS_MAP = {
    "person": 0,   # players (approximation)
    "sports ball": 1  # ball (if your model predicts it)
}

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def extract_frames(video_path: str, out_dir: Path, every_n: int = 5, max_frames: int = 2000) -> List[Path]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_paths = []
    idx, saved = 0, 0
    ensure_dir(out_dir)
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % every_n == 0:
            fp = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(fp), frame)
            frame_paths.append(fp)
            saved += 1
            if saved >= max_frames:
                break
        idx += 1
    cap.release()
    return frame_paths

def run_pseudolabels(model_weights: str, image_paths: List[Path], conf: float, labels_dir: Path):
    ensure_dir(labels_dir)
    model = YOLO(model_weights)
    # Batched prediction
    batch = [str(p) for p in image_paths]
    results = model.predict(batch, conf=conf, verbose=False)
    for img_path, res in zip(image_paths, results):
        txt_path = labels_dir / (img_path.stem + ".txt")
        lines = []
        # res.boxes has xywh or xyxy; we need normalized YOLO xywh
        imw, imh = res.orig_shape[1], res.orig_shape[0]
        for b in res.boxes:
            cls_id = int(b.cls.item())
            # Map to our CLASS_MAP if needed; else keep raw class
            # Here we keep raw class id — change if your model's classes differ
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            # Convert to normalized YOLO xywh
            cx = ((x1 + x2) / 2.0) / imw
            cy = ((y1 + y2) / 2.0) / imh
            w  = (x2 - x1) / imw
            h  = (y2 - y1) / imh
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

def make_dataset_yaml(root: Path, yaml_path: Path, names: List[str]):
    content = [
        f"path: {root.as_posix()}",
        f"train: images/train",
        f"val: images/val",
        "names:"
    ]
    for i, n in enumerate(names):
        content.append(f"  {i}: {n}")
    yaml_path.write_text("\n".join(content))

def build_from_videos(video_paths: List[str], out_root: Path, model_weights: str, conf: float = 0.4,
                      every_n: int = 5, holdout_every_k: int = 10):
    img_train = out_root / "images/train"
    img_val   = out_root / "images/val"
    lab_train = out_root / "labels/train"
    lab_val   = out_root / "labels/val"
    for p in [img_train, img_val, lab_train, lab_val]: ensure_dir(p)
    # Extract frames
    all_frames = []
    for vp in video_paths:
        frames = extract_frames(vp, out_root / f"frames_{Path(vp).stem}", every_n=every_n)
        all_frames.extend(frames)
    # Split simple holdout
    train_frames, val_frames = [], []
    for i, fp in enumerate(sorted(all_frames)):
        (val_frames if (i % holdout_every_k == 0) else train_frames).append(fp)
    # Move/copy frames into dataset folders
    for fp in train_frames:
        (img_train / fp.name).write_bytes(fp.read_bytes())
    for fp in val_frames:
        (img_val / fp.name).write_bytes(fp.read_bytes())
    # Generate pseudo-labels using current model
    run_pseudolabels(model_weights, [img_train / f.name for f in train_frames], conf, lab_train)
    run_pseudolabels(model_weights, [img_val / f.name for f in val_frames], conf, lab_val)
    # Dataset YAML
    names = [str(i) for i in range(80)]  # default COCO classes; adjust if you know your class names
    make_dataset_yaml(out_root, out_root / "dataset.yaml", names)
    return out_root / "dataset.yaml"
