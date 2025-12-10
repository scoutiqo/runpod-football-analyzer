from ultralytics import YOLO
import os

def main():
    print("🎽 TRAINING JERSEY NUMBER RECOGNIZER...")
    
    # Use YOLO Classification (v8n-cls)
    # It classifies images into "0", "1", ... "99"
    model = YOLO("yolov8n-cls.pt") 
    
    model.train(
        data="datasets/jersey_numbers",
        epochs=50,
        imgsz=64, # Small numbers don't need HD
        batch=16,
        project="runs/train",
        name="jersey_ocr",
        exist_ok=True
    )
    
    print("✅ Training Complete.")
    best = "runs/train/jersey_ocr/weights/best.pt"
    if os.path.exists(best):
        os.system(f"cp {best} models/jersey_ocr_v1.pt")
        print("   💾 Saved model: models/jersey_ocr_v1.pt")

if __name__ == "__main__":
    main()
