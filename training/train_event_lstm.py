import numpy as np
import pandas as pd
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# CONFIG
DATASET_CSV = "runs/json/event_dataset.csv"
MODEL_PATH = "models/event_lstm_model.h5"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# HYPERPARAMETERS
SEQUENCE_LENGTH = 1  # Currently we process frame-by-frame, but LSTM allows expanding this
EPOCHS = 50          # The "Learning Cycles"
BATCH_SIZE = 16

def main():
    print("🧠 Starting Deep Learning Training (LSTM)...")
    
    if not os.path.exists(DATASET_CSV):
        print("❌ Dataset not found. Run the pipeline first.")
        return

    # 1. Load Data
    df = pd.read_csv(DATASET_CSV)
    print(f"   Loaded {len(df)} training examples.")
    
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feature_cols].values
    y_raw = df['label'].values

    # 2. Preprocessing
    # Normalize features (Critical for Neural Networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode Labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    y_categorical = to_categorical(y_encoded)
    
    # Reshape for LSTM: [Samples, TimeSteps, Features]
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], SEQUENCE_LENGTH, X_scaled.shape[1]))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

    # 3. Build The Brain
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_categorical.shape[1], activation='softmax')) # Output layer

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(f"   Model Architecture: LSTM ({X_reshaped.shape[2]} features) -> {y_categorical.shape[1]} Classes")

    # 4. Train (The Learning Loop)
    print(f"   🚀 Training for {EPOCHS} Epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1 # Show progress bar
    )

    # 5. Save the Brain
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    
    print(f"✅ Deep Learning Model Saved: {MODEL_PATH}")
    
    # Report
    final_acc = history.history['val_accuracy'][-1]
    print(f"🏆 Final Validation Accuracy: {final_acc*100:.2f}%")

if __name__ == "__main__":
    main()
