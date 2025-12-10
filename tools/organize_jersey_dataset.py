import os
import csv
import shutil
import random
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("datasets/jersey_numbers")
IMG_DIR = DATA_DIR / "images"
CSV_FILE = DATA_DIR / "values.csv"

def main():
    print("🗂️ ORGANIZING JERSEY DATASET FOR YOLO...")
    
    if not CSV_FILE.exists():
        print("❌ values.csv not found.")
        return

    # Read Labels
    labels = []
    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[1].isdigit():
                labels.append((row[0], row[1])) # (filename, number)

    if not labels:
        print("❌ No valid labels found in CSV.")
        return

    print(f"   Found {len(labels)} labeled images.")

    # Create Train/Val Split (80/20)
    for split in ["train", "val"]:
        path = DATA_DIR / split
        if path.exists(): shutil.rmtree(path)
        path.mkdir(parents=True)

    count = 0
    for fname, label in tqdm(labels):
        src = IMG_DIR / fname
        
        # Some filenames in CSV might have .jpg suffix or not
        if not src.exists():
            if not fname.endswith(".jpg"): src = IMG_DIR / (fname + ".jpg")
            if not src.exists(): continue

        split = "train" if random.random() < 0.8 else "val"
        
        # Create Class Folder (e.g., train/10/)
        dest_dir = DATA_DIR / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(src, dest_dir / src.name)
        count += 1

    print(f"✅ Organized {count} images into YOLO structure.")

if __name__ == "__main__":
    main()

