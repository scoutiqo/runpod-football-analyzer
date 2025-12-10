import cv2
import json
import os
import glob
import numpy as np
from pathlib import Path

# CONFIG
DATA_DIR = Path("datasets/pitch_calibration")
IMG_DIR = DATA_DIR / "images"
LBL_DIR = DATA_DIR / "labels_json"
OUTPUT_DIR = Path("runs/viz/pitch_proofs")

# COLORS for specific points to make them easy to identify
COLORS = {
    "TL_Corner": (0, 0, 255),    # Red
    "TR_Corner": (0, 0, 255),
    "BL_Corner": (0, 0, 255),
    "BR_Corner": (0, 0, 255),
    "Center_Spot": (0, 255, 255), # Yellow
    "Penalty_Spot_Left": (255, 0, 0), # Blue
    "Penalty_Spot_Right": (255, 0, 0)
}

def main():
    print("📐 VISUALIZING PITCH CALIBRATION DATA...")
    
    if not LBL_DIR.exists():
        print("❌ No labels found. Did you run 'core/teach_pitch_calibration.py'?")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    label_files = glob.glob(f"{LBL_DIR}/*.json")
    print(f"   Found {len(label_files)} calibrated frames.")
    
    count = 0
    for lbl_file in label_files:
        try:
            # Load Data
            points = json.loads(Path(lbl_file).read_text())
            if not points: continue
            
            # Find Image
            base_name = Path(lbl_file).stem
            img_path = IMG_DIR / f"{base_name}.jpg"
            
            if not img_path.exists(): continue
            
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # Draw Points
            for name, coord in points.items():
                x_norm, y_norm = coord
                px, py = int(x_norm * w), int(y_norm * h)
                
                color = COLORS.get(name, (0, 255, 0)) # Default Green
                
                # Draw Dot
                cv2.circle(img, (px, py), 8, color, -1)
                # Draw Label
                cv2.putText(img, name, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 2)
                
            # Save Proof
            out_path = OUTPUT_DIR / f"proof_{base_name}.jpg"
            cv2.imwrite(str(out_path), img)
            count += 1
            
        except Exception as e:
            print(f"   ⚠️ Error on {lbl_file}: {e}")

    print(f"✅ Generated {count} proof images in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
