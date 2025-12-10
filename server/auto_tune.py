import os
import time
import json
import subprocess
import random
import glob
import sys
from pathlib import Path

# CONFIG
MASTER_BANK = "datasets/master_bank"
CONFIG_PATH = "models/smart_config.json"
BEST_SCORE = 0.0

# Default Parameters (The "DNA" of your system)
current_params = {
    "dist_thresh_px": 120.0,
    "dist_thresh_norm": 0.08,
    "shot_vel_thresh": 0.015,
    "min_pass_dist": 3.0,
    "duel_prox": 0.03
}

def save_config(params):
    Path(CONFIG_PATH).write_text(json.dumps(params, indent=2))

def get_validation_accuracy():
    # Runs the trainer and greps the final accuracy
    try:
        # We run the trainer in a mode that just returns score
        cmd = "python training/train_master_brain.py"
        result = subprocess.check_output(cmd, shell=True).decode()
        
        # Parse the log for "Final Validation Accuracy: 85.50%"
        for line in result.splitlines():
            if "Final Validation Accuracy" in line:
                return float(line.split(":")[-1].strip().replace("%", ""))
        return 0.0
    except:
        return 0.0

def reprocess_bank_with_params():
    # This applies the current 'smart_config.json' to the data
    # We re-run the MINER on all 10 tracks to generate new Silver Labels
    files = glob.glob("runs/json/fixed_tracks_*.json")
    print(f"   🔄 Re-simulating physics on {len(files)} matches...")
    
    for track_file in files:
        # The Miner script automatically reads models/smart_config.json
        subprocess.run(
            f"python core/events_from_tracks_pipeline.py --input \"{track_file}\"", 
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # Re-build CSV features
        subprocess.run(
            f"python training/build_event_dataset.py --input_tracks \"{track_file}\" --input_labels runs/json/silver_labels.json",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # Update Master Bank
        job_id = Path(track_file).stem.replace("fixed_tracks_", "")
        if os.path.exists("runs/json/event_dataset.csv"):
            subprocess.run(f"cp runs/json/event_dataset.csv {MASTER_BANK}/{job_id}.csv", shell=True)

def evolve():
    global BEST_SCORE, current_params
    print("🧬 STARTING EVOLUTIONARY LOOP (Auto-ML)...")
    
    # 1. Establish Baseline
    save_config(current_params)
    reprocess_bank_with_params()
    BEST_SCORE = get_validation_accuracy()
    print(f"   ⭐ Baseline AI Score: {BEST_SCORE}%")
    
    generation = 0
    while True:
        generation += 1
        print(f"\n🧪 Generation {generation}: Mutating parameters...")
        
        # A. Mutation: Change one variable by +/- 10%
        test_params = current_params.copy()
        key = random.choice(list(test_params.keys()))
        mutation = random.uniform(0.9, 1.1)
        test_params[key] *= mutation
        
        print(f"   👉 Testing {key}: {current_params[key]:.4f} -> {test_params[key]:.4f}")
        
        # B. Apply
        save_config(test_params)
        
        # C. Run Simulation
        reprocess_bank_with_params()
        
        # D. Evaluate Brain
        new_score = get_validation_accuracy()
        
        # E. Selection
        if new_score > BEST_SCORE:
            print(f"   🚀 IMPROVEMENT! {BEST_SCORE}% -> {new_score}%")
            print(f"      (The AI learned better physics rules)")
            BEST_SCORE = new_score
            current_params = test_params
            # Note: Config is already saved, so Production will use this instantly
        else:
            print(f"   📉 No gain ({new_score}%). Reverting.")
            save_config(current_params) # Revert file
            
        time.sleep(1)

if __name__ == "__main__":
    evolve()
