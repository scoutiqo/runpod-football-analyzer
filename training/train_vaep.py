import json
import numpy as np
import pandas as pd
import glob
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# CONFIG
TRACKS_DIR = "runs/json" # Where we look for past jobs
MODEL_OUT = "models/vaep_value_model.pkl"

# VAEP Lookahead Window (How far into future to check for goal)
WINDOW = 10 

def load_all_events():
    print("🔍 Aggregating Event Data for VAEP...")
    files = glob.glob(f"{TRACKS_DIR}/fixed_tracks_*.json")
    
    all_features = []
    all_labels = [] # 1 if goal occurs in next 10 events, else 0
    
    for track_file in files:
        job_id = Path(track_file).stem.replace("fixed_tracks_", "")
        # Try to find corresponding events file (we assume standard naming structure in runs/json isn't perfect history)
        # So we look for the 'final_events_viewer.json' if it was the last run, OR we skip.
        # For a proper history, we should have saved 'events_{job_id}.json'.
        # Let's try to use the 'events_dataset.csv' if available or just current 'final_events_viewer.json' 
        # for the active job to demonstrate.
        
        # FOR DEMO: We load the CURRENT final_events_viewer.json 
        # (In prod, you'd load from a database of all historical matches)
        try:
            events = json.loads(Path("runs/json/final_events_viewer.json").read_text())
        except: continue
            
        # Sort by time
        events.sort(key=lambda x: x['frame'])
        
        for i, evt in enumerate(events):
            # 1. Feature Extraction
            # What describes this action?
            label_map = {"PASS": 1, "SHOT": 2, "DUEL": 3, "TACKLE": 4, "INTERCEPTION": 5}
            type_code = label_map.get(evt['label'], 0)
            
            # Simple Spatial Features (We assume events have x,y from previous steps, or we mock it)
            # Note: final_events_viewer usually lacks x/y. We rely on timeline.
            # Let's use time and confidence as proxy features for this V1
            feat = [
                type_code,
                evt['time'],
                evt['conf'],
                evt['frame']
            ]
            
            # 2. Label Generation (The "Oracle")
            # Did a GOAL happen in the next WINDOW events?
            outcome = 0
            for k in range(1, WINDOW + 1):
                if i + k < len(events):
                    future_evt = events[i+k]
                    if future_evt['label'] == "GOAL" or future_evt['label'] == "SHOT":
                         # We value actions that lead to shots/goals
                        outcome = 1
                        break
            
            all_features.append(feat)
            all_labels.append(outcome)
            
    return np.array(all_features), np.array(all_labels)

def main():
    print("💎 STARTING VAEP VALUE TRAINING...")
    
    X, y = load_all_events()
    
    if len(X) == 0:
        print("❌ No event data found. Run the pipeline on a video first.")
        return
        
    print(f"   📊 Dataset: {len(X)} actions. Positive Value Rate: {y.mean():.2%}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model (XGBoost)
    # Scale_pos_weight handles class imbalance (goals are rare)
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        scale_pos_weight=10, 
        eval_metric='logloss'
    )
    
    print("   🚀 Training XGBoost...")
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    print("\n📊 Model Performance:")
    print(classification_report(y_test, preds))
    
    # Save
    joblib.dump(model, MODEL_OUT)
    print(f"✅ VAEP MODEL SAVED: {MODEL_OUT}")

if __name__ == "__main__":
    main()
