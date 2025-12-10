import json
import numpy as np
import os
import tensorflow as tf
import joblib
from pathlib import Path

# CONFIG
MODEL_PATH = "models/event_lstm_master.h5"
ENCODER_PATH = "models/encoder_master.pkl"
SYLLABUS_PATH = "datasets/master_bank/oracle_syllabus_deep.json"

def main():
    print("🧠 RUNNING AI IQ TEST...\n" + "="*30)

    # 1. Check Vocabulary (What does it know?)
    if Path(ENCODER_PATH).exists():
        encoder = joblib.load(ENCODER_PATH)
        classes = encoder.classes_
        print(f"📚 Vocabulary Size: {len(classes)} Concepts")
        print(f"   Top Concepts: {', '.join(classes[:5])} ... {', '.join(classes[-5:])}")
    else:
        print("❌ Encoder missing. AI has no vocabulary.")

    # 2. Check Brain Structure (How complex is it?)
    if Path(MODEL_PATH).exists():
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"🤖 Brain Architecture: {len(model.layers)} Layers")
            params = model.count_params()
            print(f"   Synapses (Parameters): {params:,}")
            
            if params < 10000:
                print("   ⚠️ Brain is too small (Underfitted).")
            elif params > 500000:
                print("   ✅ Brain is Deep (Pro Level).")
        except:
            print("❌ Could not load Brain model.")
    else:
        print("❌ Brain model missing.")

    # 3. Check Experience (How much has it seen?)
    if Path(SYLLABUS_PATH).exists():
        try:
            data = json.loads(Path(SYLLABUS_PATH).read_text())
            print(f"🎓 Training Examples: {len(data):,} situations digested.")
        except: pass

    print("="*30)

if __name__ == "__main__":
    main()
