#!/usr/bin/env python
"""
Train an event classifier on the v2 event dataset (event_dataset.csv).
INCLUDES: Feature Importance Report to validate the Physics Engine.
"""

import json
import os
import csv
from typing import List, Tuple, Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
import joblib

DATASET_CSV = "runs/json/event_dataset.csv"
DATASET_META = "runs/json/event_dataset_meta.json"
MODEL_PATH = "models/event_model.pkl"
MODEL_META_PATH = "models/event_model_meta.json"
RANDOM_SEED = 42

def _load_real_feature_names() -> List[str]:
    """Try to load human-readable feature names from metadata."""
    if os.path.exists(DATASET_META):
        try:
            with open(DATASET_META, "r") as f:
                meta = json.load(f)
            # The builder saves them in 'feature_names' but let's check structure
            # If the builder saved f_0...f_N as keys, we need the mapping.
            # Actually, the builder saves the *sorted feature keys* (f_0, f_1). 
            # We need the *definition* of those keys.
            # In v2.5 builder, we only saved 'feature_names' as [f_0, f_1...].
            # WE MISSED SAVING THE MAPPING IN THE DATASET BUILDER.
            # Fallback: We know the v2.5 schema relative to the 88 dims.
            pass
        except Exception:
            pass
    return []

def _get_feature_description(idx: int) -> str:
    """
    Manually map indices to our v2.5 schema for debugging.
    Center(22) + Mean(22) + Std(22) + DeltaWin(22) = 88 features
    """
    base_names = [
        "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_speed", "ball_accel",
        "dist_t0_c", "dist_t1_c", "min_dist_any", "is_nearest_t0",
        "min_dist_t0", "min_dist_t1", "pressure_idx", "close_cnt",
        "t0_sx", "t1_sx", "t0_sy", "t1_sy", 
        "centrality_x", "centrality_y", "pack_t0_left", "pack_t1_left"
    ]
    base_dim = len(base_names)
    
    section = idx // base_dim
    offset = idx % base_dim
    
    base_name = base_names[offset] if offset < base_dim else f"unk_{offset}"
    
    prefixes = ["Current", "Mean", "StdDev", "DeltaTotal"]
    prefix = prefixes[section] if section < len(prefixes) else "Unknown"
    
    return f"{prefix}_{base_name}"

def _load_dataset(path: str = DATASET_CSV) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows found in dataset {path}")

    feature_names = [c for c in rows[0].keys() if c.startswith("f_")]
    if not feature_names:
        raise RuntimeError("No feature columns (f_*) found in dataset.")

    # Sort them numerically just to be safe: f_0, f_1, ... f_10
    feature_names.sort(key=lambda x: int(x.split("_")[1]))

    X = np.array(
        [[float(r[c]) for c in feature_names] for r in rows],
        dtype=np.float32,
    )
    y_labels = np.array([r["label"] for r in rows], dtype=object)

    return X, y_labels, feature_names

def main():
    if not os.path.exists(DATASET_CSV):
        raise SystemExit(f"{DATASET_CSV} not found.")

    X, y_labels, raw_feature_cols = _load_dataset(DATASET_CSV)
    print(f"Dataset shape: X={X.shape}, y={y_labels.shape}")
    
    unique_labels = sorted(set(y_labels.tolist()))
    print("Labels:", unique_labels)

    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    y = np.array([label_to_idx[lbl] for lbl in y_labels], dtype=np.int64)

    # Split
    # Validation: Ensure we have enough data to even split
    if len(y) < 5:
        print("Dataset too small to split. Training on all data.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
            )
        except ValueError:
            # Fallback if stratify fails due to single class member
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=RANDOM_SEED
            )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    print("Training RandomForestClassifier...")
    clf.fit(X_train, y_train)

    # --- Feature Importance Report ---
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n" + "="*40)
    print("🏆 TOP 10 PREDICTIVE FEATURES")
    print("="*40)
    for i in range(min(10, len(indices))):
        idx = indices[i]
        col_name = raw_feature_cols[idx]
        human_name = _get_feature_description(idx) # Map f_X to "BallSpeed"
        score = importances[idx]
        print(f"{i+1:2d}. {human_name:<25} ({col_name}) : {score:.4f}")
    print("="*40 + "\n")

    # Evaluation
    if len(X_test) > 0:
        y_pred = clf.predict(X_test)
        print("Classification report (Test Set):")
        labels_sorted = sorted(set(y_test.tolist()))
        print(classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0))

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    meta = {
        "feature_names": raw_feature_cols,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "model_type": "RandomForest_v2_Pro",
    }
    with open(MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
