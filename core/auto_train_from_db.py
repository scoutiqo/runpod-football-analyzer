#!/usr/bin/env python
"""
auto_train_from_db.py (The Self-Learning Loop)

1. Connects to Supabase 'match_events'.
2. Downloads ALL events (including human corrections).
3. Re-builds the training dataset (CSV).
4. Retrains the RandomForest.
5. Deploys the new brain.
"""

import sys
import os
import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

try:
    from server.supabase_client import _get_client
    from core.event_features_v2 import build_per_frame_base_features, build_event_feature_vector
except ImportError:
    print("❌ Imports failed.")
    sys.exit(1)

DATASET_CSV = "runs/json/learned_dataset.csv"
MODEL_PATH = "models/event_model.pkl"

def fetch_ground_truth():
    sb = _get_client()
    if not sb: return []
    
    print("📥 Fetching human-verified events from Supabase...")
    # Fetch all events. In a real prod system, you'd filter by 'is_verified=true'
    res = sb.table("match_events").select("*").execute()
    data = res.data if hasattr(res, "data") else res
    
    print(f"   Found {len(data)} total events in database.")
    return data

def build_dataset_from_db(db_events):
    print("🧠 Extracting features for new training set...")
    
    # We need the tracking data associated with these events.
    # For this script to work, we need the tracks file for the specific match.
    # In production, we'd download the tracks from Supabase Storage.
    # For now, we assume we are retraining on the CURRENT local tracks.
    TRACKS_PATH = "runs/json/formatted_tracks_silver.json"
    
    if not os.path.exists(TRACKS_PATH):
        print("❌ Tracks file missing. Cannot compute features.")
        return None

    base_feats, fps, meta = build_per_frame_base_features(TRACKS_PATH)
    
    rows = []
    for evt in db_events:
        # DB columns: type, start_time (which acts as frame index)
        label = evt.get("type") or evt.get("event_type")
        frame_idx = evt.get("start_time") or evt.get("frame")
        
        # Ensure frame_idx is int
        if frame_idx is None: continue
        frame_idx = int(frame_idx)
        
        # Extract features
        if frame_idx < len(base_feats):
            vec = build_event_feature_vector(frame_idx, base_feats, fps=fps)
            row = {"label": label}
            for i, v in enumerate(vec):
                row[f"f_{i}"] = v
            rows.append(row)
            
    df = pd.DataFrame(rows)
    print(f"   Compiled {len(df)} training vectors.")
    return df

def train_new_model(df):
    print("🏋️ Training new AI model...")
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    
    # Simple split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced")
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("\n--- Model V2 Performance ---")
    print(classification_report(y_test, clf.predict(X_test), zero_division=0))
    
    # Save
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ New Brain saved to {MODEL_PATH}")

def main():
    events = fetch_ground_truth()
    if not events:
        print("No events found to learn from.")
        return

    df = build_dataset_from_db(events)
    if df is not None and not df.empty:
        train_new_model(df)

if __name__ == "__main__":
    main()
