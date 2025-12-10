import time
import subprocess
import sys

python_exe = sys.executable

def main():
    print("♾️ STARTING UNIFIED LEARNING LOOP (BRAIN + EYES)...")
    print("   Press Ctrl+C to stop.")
    
    cycle = 1
    while True:
        print(f"\n🔄 CYCLE {cycle}: Expanding Knowledge Base...")
        
        # 1. MINE EVENTS (The Analyst)
        print("   [1/4] Mining Tactical Events...")
        subprocess.run(f"{python_exe} core/active_learning_loop.py", shell=True)
        
        # 2. MINE GEOMETRY (The Surveyor)
        print("   [2/4] Mining Pitch Geometry...")
        subprocess.run(f"{python_exe} core/teach_pitch_calibration.py", shell=True)
        
        # 3. TRAIN BRAIN (LSTM)
        print(f"   [3/4] Retraining Master Brain (Events)...")
        subprocess.run(f"{python_exe} training/train_master_brain.py", shell=True)

        # 4. TRAIN EYES (YOLO Pose) - Run every 5 cycles to save time
        # (Training YOLO is heavier than LSTM)
        if cycle % 5 == 0:
            print(f"   [4/4] Retraining Pitch Calibrator (YOLO)...")
            subprocess.run(f"{python_exe} core/train_calibration_model.py", shell=True)
        else:
            print(f"   [4/4] Skipping Calibrator Training (Scheduled for Cycle {cycle + (5 - cycle%5)})")
        
        print("💤 Sleeping 30s...")
        time.sleep(30)
        cycle += 1

if __name__ == "__main__":
    main()
