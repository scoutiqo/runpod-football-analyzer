import subprocess
import json
import random
from pathlib import Path

# TARGET VIDEO (Use one robust example)
TEST_VIDEO = "viewer/test_short.mp4"
TEST_ID = "auto_tuning_v1"

# PARAMETERS TO TUNE
# format: (file, line_marker, min, max)
# We will use a simple config dict approach for now to avoid complex parsing
CONFIG_PATH = "core/auto_tune.json"

def load_config():
    if not Path(CONFIG_PATH).exists():
        return {
            "conf_thresh": 0.1,
            "team_cluster_weight": 3.0,
            "event_dist_thresh": 0.08
        }
    return json.loads(Path(CONFIG_PATH).read_text())

def run_pipeline(config):
    # 1. Update config files/args based on 'config'
    # For simplicity, we pass these as CLI args to a modified pipeline runner
    # But here, we will simulate it by running the Pro Pipeline and checking the output
    
    print(f"🧬 Testing Gene: {config}")
    
    cmd = [
        "python", "core/run_pro_pipeline.py",
        "--video", TEST_VIDEO,
        "--match_id", TEST_ID,
        "--save_tracks", "runs/json/tracks_evolve.json"
    ]
    
    # We need to update run_pro_pipeline to accept these tuning params
    # or we write them to a temp file that the scripts read.
    Path("params.json").write_text(json.dumps(config))
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 2. Evaluate Result
        # Read output
        tracks = json.loads(Path("runs/json/tracks_evolve.json").read_text())
        
        # Score 1: Detection Count (More is better, up to a limit)
        n_players = sum(len(f['players']) for f in tracks['frames'])
        score_detection = min(n_players, 50000) / 50000
        
        # Score 2: Team Balance (A vs B ratio should be close to 1)
        count_a = 0
        count_b = 0
        for f in tracks['frames']:
            for p in f['players']:
                if p['team'] == 'A': count_a += 1
                if p['team'] == 'B': count_b += 1
                
        total = count_a + count_b
        if total == 0: return 0
        
        balance = 1.0 - abs(count_a - count_b) / total
        
        # Total Score
        fitness = (score_detection * 0.3) + (balance * 0.7)
        print(f"   💪 Fitness: {fitness:.4f} (Bal: {balance:.2f})")
        return fitness
        
    except Exception as e:
        print(f"   💀 Died: {e}")
        return 0

def evolve():
    print("🦠 Starting Evolution Loop...")
    best_score = 0
    best_config = load_config()
    
    for generation in range(20): # Run 20 experiments
        # Mutate
        current_config = best_config.copy()
        mutation_factor = random.uniform(0.8, 1.2)
        
        current_config['team_cluster_weight'] *= mutation_factor
        
        score = run_pipeline(current_config)
        
        if score > best_score:
            print(f"   🎉 NEW BEST! Score: {score:.4f}")
            best_score = score
            best_config = current_config
            Path(CONFIG_PATH).write_text(json.dumps(best_config, indent=2))
        else:
            print("   📉 Degressed.")

if __name__ == "__main__":
    evolve()
