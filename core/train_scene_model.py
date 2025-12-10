from ultralytics import YOLO

def main():
    print("🧠 TRAINING SCENE DIRECTOR AI...")
    
    # Load a pre-trained classification model
    # yolov8n-cls is incredibly fast
    model = YOLO("yolov8n-cls.pt") 
    
    model.train(
        data="datasets/scene_classification",
        epochs=30,
        imgsz=224, # Small image size is fine for scene detection
        batch=16,
        project="runs/train",
        name="scene_director",
        exist_ok=True
    )
    
    print("✅ Director Trained. Saved to models/scene_director_v1.pt")
    # Copy to models
    import shutil
    shutil.copy("runs/train/scene_director/weights/best.pt", "models/scene_director_v1.pt")

if __name__ == "__main__":
    main()
