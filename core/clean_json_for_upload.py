import json
import math
import sys
from pathlib import Path

def sanitize(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj): return 0.0
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj

def main():
    print("🧹 SANITIZING JSON FILES...")
    base = Path("runs/json")
    files = ["tracks.json", "events.json", "formation.json", "advanced_metrics.json", "possession_chains.json"]
    
    for fname in files:
        fpath = base / fname
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text())
                clean_data = sanitize(data)
                fpath.write_text(json.dumps(clean_data))
                print(f"   ✅ Cleaned {fname}")
            except:
                print(f"   ⚠️ Could not clean {fname}")

if __name__ == "__main__":
    main()
