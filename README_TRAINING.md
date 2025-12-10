# 🚀 ScoutIQO Training System

## **Overview**
Real YOLO training with pseudo-labeling from your football videos. No fake data - everything is generated from actual video analysis.

## **How It Works**

### **1. Pseudo-Labeling Process**
- Extracts frames from your videos (every 5th frame by default)
- Runs your current YOLO model on these frames
- Generates YOLO-format labels from actual detections
- Creates train/val split automatically

### **2. Real Training**
- Uses Ultralytics YOLO trainer
- Real epochs with real loss curves
- Saves checkpoints after each epoch
- Streams progress to logs

### **3. Supabase Integration**
- Uploads all training artifacts
- Stores checkpoints, logs, and results
- Real-time progress tracking

## **Usage**

### **Via RunPod Handler**
```json
{
  "mode": "train",
  "run_id": "run-001",
  "model_base": "yolov8n.pt",
  "epochs": 10,
  "video_urls": ["signed-url1.mp4", "signed-url2.mp4"],
  "pseudolabel_conf": 0.4,
  "pseudolabel_every_n": 5
}
```

### **Local Training**
```bash
python -m training.train \
  --model_base yolov8n.pt \
  --out_dir ./training_output \
  --epochs 10 \
  --videos_json '["video1.mp4", "video2.mp4"]'
```

## **Output Structure**
```
training_output/
├── logs/
│   └── train.jsonl          # Epoch-by-epoch metrics
├── checkpoints/
│   ├── epoch_01_last.pt     # Checkpoints
│   ├── epoch_02_last.pt
│   └── best.pt              # Best model
├── pseudolabel_dataset/
│   ├── images/train/        # Training images
│   ├── images/val/          # Validation images
│   ├── labels/train/        # Pseudo-labels
│   ├── labels/val/
│   └── dataset.yaml         # YOLO dataset config
└── ultra_run/               # Ultralytics output
```

## **Real Metrics**
- **TRAIN_EPOCH**: Real epoch progress
- **CHECKPOINT**: Real checkpoint saves
- **RESULT_DIR**: Final output location
- **Loss curves**: Real training loss
- **Learning rate**: Real LR scheduling

## **No Simulations**
Everything is generated from your actual football videos using real YOLO detections. Quality improves as you iterate and fine-tune.
