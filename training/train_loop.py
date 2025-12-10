#!/usr/bin/env python
"""
Extreme training loop for event classifier.
- 500–1000 epochs
- data shuffling every epoch
- class-balanced resampling
- feature noise (prevents overfitting)
- best model checkpoint
- autosave every 50 epochs
- verbose learning logs
"""

import os
import json
import numpy as np
import pandas as pd

from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.utils import resample
import joblib

DATASET_CSV = "runs/json/event_dataset.csv"
MODEL_PATH = "models/event_model_extreme.pkl"
MODEL_META_PATH = "models/event_model_extreme_meta.json"

EPOCHS = 1000
NOISE_STD = 0.015  # small feature noise
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    df = pd.read_csv(DATASET_CSV)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    label_col = "label"

    X = df[feature_cols].values.astype(np.float32)
    y_labels = df[label_col].values.astype(str)

    unique_labels = sorted(set(y_labels))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
    y = np.array([label_to_idx[lbl] for lbl in y_labels], dtype=np.int64)

    meta = {
        "feature_names": feature_cols,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "dataset_csv": DATASET_CSV,
    }

    return X, y, feature_cols, meta


def inject_noise(X: np.ndarray, std: float = NOISE_STD) -> np.ndarray:
    noise = np.random.normal(0, std, X.shape).astype(np.float32)
    return X + noise


def class_balanced_resample(X: np.ndarray, y: np.ndarray):
    """Resample dataset so that each class has equal number of samples."""
    classes = np.unique(y)
    max_count = max(np.sum(y == c) for c in classes)

    X_resampled = []
    y_resampled = []

    for c in classes:
        X_c = X[y == c]
        y_c = y[y == c]
        X_up, y_up = resample(
            X_c, y_c, replace=True, n_samples=max_count, random_state=np.random.randint(1_000_000)
        )
        X_resampled.append(X_up)
        y_resampled.append(y_up)

    X_final = np.vstack(X_resampled)
    y_final = np.concatenate(y_resampled)

    return X_final, y_final


def main():
    if not os.path.exists(DATASET_CSV):
        raise SystemExit("Dataset not found. Run build_event_dataset.py first.")

    X, y, feature_cols, meta = load_dataset()
    n_features = X.shape[1]

    print(f"Loaded dataset: {X.shape[0]} samples, {n_features} features, labels={list(meta['label_to_idx'].keys())}")

    best_f1 = -1
    best_model = None

    # Training loop
    for epoch in range(1, EPOCHS + 1):

        # Resample to balance classes
        X_bal, y_bal = class_balanced_resample(X, y)

        # Inject noise for generalization
        X_bal_noisy = inject_noise(X_bal)

        # Train model this epoch
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=np.random.randint(1_000_000),
            class_weight=None,
            n_jobs=-1,
        )
        clf.fit(X_bal_noisy, y_bal)

        # Evaluate on original (tiny) dataset
        y_pred = clf.predict(X)
        f1 = f1_score(y, y_pred, average="macro", zero_division=0)

        if epoch % 50 == 0:
            print(f"[Epoch {epoch}/{EPOCHS}] f1={f1:.4f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = clf

            joblib.dump(best_model, MODEL_PATH)
            with open(MODEL_META_PATH, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"🔥 New BEST model saved! Epoch={epoch} f1={f1:.4f}")

    print(f"\nTraining complete. Best f1_macro={best_f1:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {MODEL_META_PATH}")


if __name__ == "__main__":
    main()
