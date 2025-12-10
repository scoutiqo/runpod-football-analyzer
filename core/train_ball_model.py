from ultralytics import YOLO
import os

def main():
    print("🏋️ TRAINING SPECIALIST BALL MODEL...")
    model = YOLO("yolov8n.pt") # Start small and fast
    
    model.train(
        data="datasets/ball_vision/data.yaml",
        epochs=50,
        imgsz=1280,
        batch=8,
        project="runs/train",
        name="ball_specialist",
        exist_ok=True
    )
    
    best = "runs/train/ball_specialist/weights/best.pt"
    if os.path.exists(best):
        os.system(f"cp {best} models/football_ball_v1.pt")
        print(f"✅ BALL MODEL SAVED: models/football_ball_v1.pt")

if __name__ == "__main__":
    main()
