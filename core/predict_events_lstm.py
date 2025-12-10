import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#!/usr/bin/env python
import json
import os
import numpy as np
import pandas as pd
import joblib
import argparse
from pathlib import Path
from tensorflow.keras.models import load_model

# IMPORTS
import sys
sys.path.append(os.getcwd())
from core.event_features_v2 import build_per_frame_base_features, build_event_feature_vector

# CONFIG
MODEL_PATH = "models/event_lstm_master.h5"
SCALER_PATH = "models/scaler_master.pkl"
ENCODER_PATH = "models/encoder_master.pkl"
SEQUENCE_LENGTH = 5 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True, help="Input tracks JSON")
    parser.add_argument("--output", default="runs/json/predicted_events_learned.json")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    print(f"🧠 Loading Master Brain: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
    except Exception as e:
        print(f"❌ Failed to load model assets: {e}")
        return
    
    # 1. Build Base Features (N, 22)
    print(f"   Extracting raw physics from: {args.tracks}")
    try:
        base_feats, fps, _ = build_per_frame_base_features(args.tracks)
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return

    # 2. Expand Features (N, 22) -> (N, 88)
    # The model expects the calculated deltas/means, not just raw coords
    print(f"   Expanding feature vectors (22 -> 88 dims)...")
    expanded_feats = []
    
    for i in range(len(base_feats)):
        # This function turns the single frame + context into the 88 features
        vec = build_event_feature_vector(i, base_feats, fps=fps)
        expanded_feats.append(vec)
        
    X_88 = np.array(expanded_feats) # Shape: (N, 88)
    
    # 3. Scale (Now dimensions match!)
    try:
        X_scaled = scaler.transform(X_88)
    except ValueError as e:
        print(f"❌ Scaling Error: {e}")
        print(f"   Expected {scaler.n_features_in_} features, got {X_88.shape[1]}")
        return

    # 4. Sequence Generation (Sliding Window for LSTM)
    # Input: (N, 88) -> Output: (Samples, 5, 88)
    X_seq = []
    valid_indices = []
    
    for i in range(SEQUENCE_LENGTH, len(X_scaled)):
        window = X_scaled[i-SEQUENCE_LENGTH : i] # (5, 88)
        X_seq.append(window)
        valid_indices.append(i)
        
    if not X_seq:
        print("⚠️ Not enough frames for sequence analysis.")
        return

    X_input = np.array(X_seq)
    
    # 5. Predict
    print(f"   Running Inference on {len(X_input)} sequences...")
    probs = model.predict(X_input, verbose=0)
    
    # 6. Decode
    pred_indices = np.argmax(probs, axis=1)
    labels = encoder.inverse_transform(pred_indices)
    max_probs = np.max(probs, axis=1)
    
    # 7. Save Results
    all_preds = []
    for k, idx in enumerate(valid_indices):
        all_preds.append({
            "frame": idx,
            "label": labels[k],
            "prob": float(max_probs[k])
        })
        
    Path(args.output).write_text(json.dumps(all_preds, indent=2))
    print(f"✅ Saved {len(all_preds)} predictions to {args.output}")

if __name__ == "__main__":
    main()
