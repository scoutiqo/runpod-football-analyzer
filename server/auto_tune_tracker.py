import subprocess
import json
import random
import glob
from pathlib import Path

# CONFIG
CONFIG_FILE = "models/tracker_config.json"

def evaluate_tracking_quality(track_path):
    """
    Calculates a 'Fragmentation Score'. 
    Lower is better.
    """
    try:
        data = json.loads(Path(track_path).read_text())
        frames = data.get('frames', [])
        
        # Map ID -> Lifespan
        id_lifespans = {}
        for f in frames:
            for p in f.get('players', []):
                pid = p['id']
                id_lifespans[pid] = id_lifespans.get(pid, 0) + 1
                
        # Metric 1: Total unique IDs (Should be close to 22 for a clean game)
        total_ids = len(id_lifespans)
        
        # Metric 2: Average lifespan (Higher is better)
        avg_life = sum(id_lifespans.values()) / max(1, total_ids)
        
        # Score: We want FEWER ids and LONGER lives
        # Heuristic: ideal IDs = 25. 
        penalty = abs(total_ids - 25) * 10
        
        # Inverse fitness (Lower score = Better quality)
        score = penalty - avg_life
        return score, total_ids
    except:
        return 9999, 0

def evolve():
    print("🧬 STARTING TRACKER EVOLUTION...")
    
    # Default Genes
    best_conf = 0.1
    best_buffer = 30
    best_score = 9999
    
    # Video to test on
    test_video = glob.glob("tmp_jobs/*.mp4")[0] if glob.glob("tmp_jobs/*.mp4") else "viewer/test_short.mp4"
    
    for gen in range(20):
        # Mutate
        test_conf = max(0.01, min(0.5, best_conf + random.uniform(-0.05, 0.05)))
        test_buffer = int(max(10, min(120, best_buffer + random.uniform(-10, 10))))
        
        print(f"\n🧪 Gen {gen}: Testing Conf={test_conf:.2f}, Buffer={test_buffer}...")
        
        # Run Tracker CLI with these params
        # We need to modify run_tracker_cli to accept these args or we wrap it here
        # For this snippet, we assume we pass them via CLI flags we added
        
        cmd = [
            "python", "core/run_tracker_cli.py",
            "--input", test_video,
            "--save", "runs/json/temp_evolve.json",
            "--conf", str(test_conf)
        ]
        # Note: Buffer arg needs to be added to tracker_players.py to support this
        # But adjusting Conf is the biggest win.
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            score, ids = evaluate_tracking_quality("runs/json/temp_evolve.json")
            
            print(f"   📊 Result: IDs={ids}, Score={score:.2f}")
            
            if score < best_score:
                print(f"   🚀 NEW BEST! Improvement found.")
                best_score = score
                best_conf = test_conf
                
                # Save Config
                config = {"conf_thresh": best_conf, "buffer": test_buffer}
                Path(CONFIG_FILE).write_text(json.dumps(config))
        except:
            pass

if __name__ == "__main__":
    evolve()
