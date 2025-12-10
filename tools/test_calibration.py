from ultralytics import YOLO
import cv2
import glob
import os
import random
from pathlib import Path

# CONFIG
MODEL_PATH = "models/pitch_calibration_v1.pt"
TEST_IMG_DIR = "datasets/pitch_calibration/images"
OUTPUT_DIR = "runs/viz/calibration_test"

def main():
    print("🧪 TESTING PITCH CALIBRATION MODEL...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    images = glob.glob(f"{TEST_IMG_DIR}/*.jpg")
    random.shuffle(images)
    
    print(f"   Running inference on 10 random images...")
    for img_path in images[:10]:
        results = model(img_path)
        for r in results:
            # Plot keypoints
            im_array = r.plot(kpt_radius=5, boxes=False) 
            out_path = f"{OUTPUT_DIR}/pred_{os.path.basename(img_path)}"
            cv2.imwrite(out_path, im_array)
            print(f"   ✅ Saved: {out_path}")

    print(f"\n🎉 Check {OUTPUT_DIR} to see if the AI understands the pitch.")

if __name__ == "__main__":
    main()
