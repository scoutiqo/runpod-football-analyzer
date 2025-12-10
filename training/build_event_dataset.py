import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path

# Import Feature Logic
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from core.event_features_v2 import build_per_frame_base_features, build_event_feature_vector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tracks", required=True)
    parser.add_argument("--input_labels", required=True)
    parser.add_argument("--output_csv", default="runs/json/event_dataset.csv")
    args = parser.parse_args()

    print(f"🏗️ Building Dataset from: {args.input_tracks}")
    
    if not Path(args.input_labels).exists():
        print("❌ Labels file missing. Run miner first.")
        return

    # Load Labels
    labels = json.loads(Path(args.input_labels).read_text())
    print(f"   Loaded {len(labels)} events.")
    
    # Build Features
    base_feats, fps, _ = build_per_frame_base_features(args.input_tracks)
    print(f"   Features ready: {len(base_feats)} frames.") # <--- CHECK THIS NUMBER IN LOGS

    rows = []
    for evt in labels:
        f_idx = evt['frame']
        lbl = evt['label']
        
        if f_idx < len(base_feats):
            vec = build_event_feature_vector(f_idx, base_feats, fps=fps)
            row = {f"f_{i}": v for i, v in enumerate(vec)}
            row['label'] = lbl
            row['frame'] = f_idx
            rows.append(row)
            
    if not rows:
        print("⚠️ No valid rows created.")
        return
        
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"✅ Success! Wrote {len(rows)} samples to {args.output_csv}")

if __name__ == "__main__":
    main()
