#!/usr/bin/env python
"""
predict_events_from_tracks.py (v2.5 Pro - Fixed)

Runs the Event Foundation Model on the full track sequence.
Uses the 'Frankenstein' merged tracks to ensure features (Team Centroids, Packing) are valid.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

# Import the Pro-Level Feature Extractor
from event_features_v2 import build_per_frame_base_features, build_event_feature_vector

MODEL_PATH = "models/event_model.pkl"
MODEL_META_PATH = "models/event_model_meta.json"
# IMPORTANT: Point to the merged file with Teams + Ball + Players
DEFAULT_TRACKS_PATH = "runs/json/formatted_tracks_silver.json" 
OUT_PREDS = "runs/json/predicted_events_learned.json"

def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run train_event_model.py first.")
    
    # Determine which tracks file to use
    tracks_path = DEFAULT_TRACKS_PATH
    
    if not os.path.exists(tracks_path):
        print(f"Warning: {tracks_path} not found. Trying simple tracks...")
        fallback = "runs/json/tracks_players_ball_simple.json"
        if os.path.exists(fallback):
            tracks_path = fallback
            print(f"-> Using fallback: {tracks_path}")
        else:
            raise SystemExit("No tracks file found to run inference on.")

    print(f"Loading model from {MODEL_PATH} ...")
    clf = joblib.load(MODEL_PATH)

    meta = json.loads(Path(MODEL_META_PATH).read_text(encoding="utf-8"))
    idx_to_label = {int(k): v for k, v in meta.get("idx_to_label", {}).items()}
    
    # 1. Build Features
    print(f"Building features from {tracks_path} ...")
    # This automatically uses the v2.5 logic (22 base dims -> 88 vector dims)
    base_feats, fps, _ = build_per_frame_base_features(tracks_path)
    n_frames, base_dim = base_feats.shape
    print(f"Frames: {n_frames}, Base Feature Dim: {base_dim}")

    preds_out: List[Dict[str, Any]] = []

    # 2. Inference Loop
    print("Running inference on all frames...")
    
    for frame_idx in range(n_frames):
        # Generate the 88-dim vector
        feat_vec = build_event_feature_vector(frame_idx, base_feats, fps=fps, window_seconds=0.6)
        
        # Reshape for sklearn (1, 88)
        X = feat_vec.reshape(1, -1)
        
        # Predict
        proba = clf.predict_proba(X)[0]
        label_idx = int(np.argmax(proba))
        label_name = idx_to_label.get(label_idx, str(label_idx))
        best_prob = float(proba[label_idx])

        preds_out.append({
            "frame": frame_idx,
            "label": label_name,
            "prob": best_prob,
            # detailed probs for debug
            "probs": {idx_to_label.get(i, str(i)): float(p) for i, p in enumerate(proba)}
        })

    # 3. Save
    os.makedirs(os.path.dirname(OUT_PREDS), exist_ok=True)
    Path(OUT_PREDS).write_text(json.dumps(preds_out, indent=2), encoding="utf-8")
    print(f"✅ Wrote {len(preds_out)} predictions to {OUT_PREDS}")

    # 4. Report Top Confident Events
    # Filter for interesting events with high confidence
    interesting = [p for p in preds_out if p["prob"] > 0.65]
    interesting.sort(key=lambda x: x["prob"], reverse=True)
    
    print("\n🔍 Sample High-Confidence Predictions:")
    print(f"{'Frame':<8} {'Label':<12} {'Conf':<6} {'Other Probs'}")
    print("-" * 60)
    for p in interesting[:15]:
        other = ", ".join([f"{k}:{v:.2f}" for k,v in p["probs"].items() if v > 0.1])
        print(f"{p['frame']:<8d} {p['label']:<12} {p['prob']:.2f}   {other}")

if __name__ == "__main__":
    main()
