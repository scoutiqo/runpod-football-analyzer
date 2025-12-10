import json
import math
from pathlib import Path

# CONFIG
INPUT_TRACKS = "runs/json/tracks.json"
OUTPUT_TRACKS = "runs/json/tracks_vis.json" # Optimized file

def main():
    print("📦 OPTIMIZING DATA FOR FRONTEND...")
    if not Path(INPUT_TRACKS).exists(): return

    data = json.loads(Path(INPUT_TRACKS).read_text())
    
    # 1. Downsample FPS (25 -> 5)
    # We only need 5 updates per second for a smooth-looking map
    data['frames'] = data['frames'][::5]
    data['fps'] = 5 
    
    # 2. Round Coordinates (Drastic file size reduction)
    for f in data['frames']:
        # Remove heavy debug data
        if 'homography' in f: del f['homography']
        
        for p in f['players']:
            p['x'] = round(p['x'], 3)
            p['y'] = round(p['y'], 3)
            # Remove raw meters if we aren't using them for physics on frontend
            # or round them aggressively
            if 'x_m' in p: p['x_m'] = round(p['x_m'], 1)
            if 'y_m' in p: p['y_m'] = round(p['y_m'], 1)

    Path(OUTPUT_TRACKS).write_text(json.dumps(data, separators=(',', ':')))
    print(f"✅ Reduced file size. Saved to {OUTPUT_TRACKS}")

if __name__ == "__main__":
    main()
