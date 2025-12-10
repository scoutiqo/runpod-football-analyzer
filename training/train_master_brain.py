import pandas as pd
import numpy as np
import glob
import os
import json
import joblib
import sys
from pathlib import Path
import tensorflow as tf

# GPU Memory Limit
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try: tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    except: pass

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from collections import Counter

# CONFIG
# We prioritize the CLEAN dataset if it exists
CLEAN_DATASET = "datasets/master_bank/clean_master_dataset.csv"
MASTER_DIR = "datasets/master_bank"
MODEL_PATH = "models/event_lstm_master.h5"
SCALER_PATH = "models/scaler_master.pkl"
ENCODER_PATH = "models/encoder_master.pkl"

SEQUENCE_LENGTH = 5 
EPOCHS = 50
BATCH_SIZE = 16

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def main():
    print("🧠 STARTING MASTER BRAIN TRAINING...")
    
    # 1. LOAD DATA
    if os.path.exists(CLEAN_DATASET):
        print(f"   ✅ Using Clean Dataset: {CLEAN_DATASET}")
        full_df = pd.read_csv(CLEAN_DATASET)
    else:
        print("   ⚠️ Clean dataset not found. Falling back to raw files (Warning: May contain duplicates).")
        files = glob.glob(f"{MASTER_DIR}/*.csv")
        files = [f for f in files if "clean" not in f and "augmented" not in f]
        
        if not files:
            print("❌ No training data found.")
            return
            
        dfs = []
        for f in files:
            try: dfs.append(pd.read_csv(f))
            except: pass
        full_df = pd.concat(dfs, ignore_index=True)

    print(f"   📊 Total Samples: {len(full_df)}")
    
    # 2. PREPROCESSING
    # Filter Unknowns
    full_df = full_df[full_df['label'] != 'unknown']
    
    feature_cols = [c for c in full_df.columns if c.startswith("f_")]
    X_raw = full_df[feature_cols].values
    y_raw = full_df['label'].values
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Stats
    counts = Counter(y_raw)
    print(f"   Events: {len(counts)} classes found.")
    
    # 3. ENCODING
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    y_cat = to_categorical(y_encoded)
    
    # 4. SCALING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 5. SEQUENCING
    X_seq, y_seq = create_sequences(X_scaled, y_cat, SEQUENCE_LENGTH)
    
    if len(X_seq) == 0:
        print("❌ Not enough data for sequences.")
        return

    # 6. CLASS WEIGHTS
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weights = dict(enumerate(weights))

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # 7. MODEL
    model = Sequential([
        LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        BatchNormalization(),
        Dropout(0.4),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(y_cat.shape[1], activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print("\n🚀 IGNITION: Training...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
              validation_data=(X_test, y_test), class_weight=class_weights, verbose=1)
    
    # 8. SAVE
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"\n✅ BRAIN SAVED: {MODEL_PATH}")

if __name__ == "__main__":
    main()
