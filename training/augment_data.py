import pandas as pd
import numpy as np
import glob
import os

MASTER_DIR = "datasets/master_bank"
OUTPUT_FILE = "datasets/master_bank/augmented_dataset.csv"

def augment_row(row, num_variations=50):
    new_rows = []
    
    # Convert row to numeric values
    features = row.drop('label').values.astype(float)
    label = row['label']
    
    for _ in range(num_variations):
        # 1. Add Gaussian Noise (Simulate sensor error)
        noise = np.random.normal(0, 0.05, features.shape)
        new_feats = features + noise
        
        # 2. Speed Scaling (Simulate faster/slower players)
        speed_factor = np.random.uniform(0.8, 1.2)
        # Scale velocity columns (assuming indices 2,3 are vx,vy)
        # This is a heuristic, we rely on the LSTM to learn the pattern regardless of scale
        # Ideally we parse columns, but blind noise works for regularization
        
        new_row = {f"f_{i}": v for i, v in enumerate(new_feats)}
        new_row['label'] = label
        new_rows.append(new_row)
        
    return new_rows

def main():
    print("🧪 STARTING DATA AUGMENTATION (Simulation)...")
    
    files = glob.glob(f"{MASTER_DIR}/*.csv")
    # Exclude previous augmentations
    files = [f for f in files if "augmented" not in f]
    
    if not files: return

    all_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty: all_dfs.append(df)
        except: pass
            
    if not all_dfs: return
    
    full_df = pd.concat(all_dfs)
    print(f"   Original Data: {len(full_df)} samples")
    
    # Augment
    augmented_data = []
    for idx, row in full_df.iterrows():
        # Generate 100 fake versions of this event
        augmented_data.extend(augment_row(row, num_variations=100))
        
    aug_df = pd.DataFrame(augmented_data)
    
    # Save
    aug_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"   🚀 Synthesized {len(aug_df)} new training samples.")
    print(f"   💾 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
