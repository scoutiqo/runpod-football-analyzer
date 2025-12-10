import json
import os
import time
from pathlib import Path

MODEL = Path("models/event_lstm_master.h5")
SYLLABUS = Path("datasets/master_bank/oracle_syllabus_deep.json")
JERSEY_DATA = Path("datasets/jersey_numbers/images")

def main():
    print("\n🧠 LEARNING LOOP AUDIT\n" + "="*30)
    
    # 1. BRAIN HEALTH
    if MODEL.exists():
        mod_time = time.ctime(os.path.getmtime(MODEL))
        size_mb = os.path.getsize(MODEL) / (1024*1024)
        print(f"🤖 Brain Model: {size_mb:.2f} MB")
        print(f"⌚ Last Updated: {mod_time}")
    else:
        print("❌ Brain Model MISSING.")

    # 2. KNOWLEDGE BASE
    if SYLLABUS.exists():
        try:
            data = json.loads(SYLLABUS.read_text())
            print(f"📚 Learned Concepts: {len(data)} unique examples")
        except:
            print("⚠️ Syllabus file is corrupt.")
    
    # 3. VISION (Jersey Numbers)
    if JERSEY_DATA.exists():
        count = len(list(JERSEY_DATA.glob("*.jpg")))
        print(f"🎽 Jersey Samples Mined: {count}")
    else:
        print("❌ No Jersey data mined yet.")

if __name__ == "__main__":
    main()
