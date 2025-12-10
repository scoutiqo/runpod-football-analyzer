from ultralytics import YOLO
import os
import shutil
import json
import yaml
from pathlib import Path
import random

# CONFIG
DATASET_DIR = Path("datasets/pitch_calibration")
RAW_IMG_DIR = DATASET_DIR / "images"
RAW_JSON_DIR = DATASET_DIR / "labels_json"

# YOLO STRUCTURE
TRAIN_IMG = DATASET_DIR / "train/images"
TRAIN_LBL = DATASET_DIR / "train/labels"
VAL_IMG = DATASET_DIR / "val/images"
VAL_LBL = DATASET_DIR / "val/labels"

KEYPOINTS = [
    "TL_Corner", "TR_Corner", "BR_Corner", "BL_Corner",
    "Center_Circle_Top", "Center_Circle_Bottom", "Center_Spot",
    "Penalty_Spot_Left", "Penalty_Spot_Right",
    "Box_TL_Left", "Box_BL_Left", "Box_TR_Right", "Box_BR_Right"
]

def setup_yolo_dirs():
    for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        d.mkdir(parents=True, exist_ok=True)

def convert_and_split():
    print("🔄 Converting JSON to YOLO Pose Format & Splitting...")
    
    json_files = list(RAW_JSON_DIR.glob("*.json"))
    if not json_files:
        print("❌ No JSON labels found. Run the miner first.")
        return False
        
    random.shuffle(json_files)
    
    # 80/20 Split
    split_idx = int(len(json_files) * 0.8)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]
    
    def process_set(files, img_dest, lbl_dest):
        for jf in files:
            # Load JSON
            try:
                data = json.loads(jf.read_text())
            except: continue
            
            if not data: continue
            
            # Build Keypoints Line
            # Format: <class> <x_center> <y_center> <width> <height> <px1> <py1> <vis1> ...
            # Class 0 (Pitch), Box = Whole Image
            
            kpts = []
            has_visible = False
            for kp_name in KEYPOINTS:
                if kp_name in data:
                    x, y = data[kp_name]
                    kpts.extend([x, y, 2]) # 2 = Visible
                    has_visible = True
                else:
                    kpts.extend([0, 0, 0]) # 0 = Missing
            
            if not has_visible: continue

            line = "0 0.5 0.5 1.0 1.0 " + " ".join(map(str, kpts))
            
            # Save Label
            (lbl_dest / f"{jf.stem}.txt").write_text(line)
            
            # Copy Image
            src_img = RAW_IMG_DIR / f"{jf.stem}.jpg"
            if src_img.exists():
                shutil.copy(src_img, img_dest / src_img.name)

    process_set(train_files, TRAIN_IMG, TRAIN_LBL)
    process_set(val_files, VAL_IMG, VAL_LBL)
    
    print(f"   ✅ Train: {len(list(TRAIN_LBL.glob('*.txt')))} samples")
    print(f"   ✅ Val:   {len(list(VAL_LBL.glob('*.txt')))} samples")
    return True

def main():
    setup_yolo_dirs()
    if not convert_and_split(): return
    
    print("📐 STARTING CALIBRATION MODEL TRAINING...")
    
    # Create Data YAML
    yaml_content = {
        "path": str(DATASET_DIR.absolute()),
        "train": "train/images",
        "val": "val/images",
        "kpt_shape": [13, 3],
        "names": {0: "pitch"}
    }
    
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
        
    # Train
    model = YOLO("yolov8n-pose.pt")
    
    model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=8,
        project="runs/train",
        name="pitch_calibrator",
        exist_ok=True
    )
    
    # Save
    best = "runs/train/pitch_calibrator/weights/best.pt"
    if os.path.exists(best):
        os.system(f"cp {best} models/pitch_calibration_v1.pt")
        print(f"✅ CALIBRATION MODEL SAVED: models/pitch_calibration_v1.pt")

if __name__ == "__main__":
    main()
