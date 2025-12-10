import json
import numpy as np
from pathlib import Path
import pandas as pd

# CONFIG
INPUT_FILE = Path("runs/json/formatted_tracks_silver.json")
OUTPUT_FILE = Path("runs/json/formatted_tracks_silver.json")
MAX_INTERPOLATION_GAP = 50 # Max frames to bridge (2 seconds)
ROLLING_AVG_WINDOW = 5

def smooth_ball_trajectory():
    print("⚾ Starting Ball Stabilization...")
    
    tracks_data = json.loads(INPUT_FILE.read_text())
    frames = tracks_data.get('frames', [])
    fps = tracks_data.get('fps', 25.0)

    # 1. Extract Ball Trajectory into DataFrame
    ball_rows = []
    for i, f in enumerate(frames):
        b = f.get('ball')
        ball_rows.append({
            'frame': i,
            'x': b.get('x') if b else np.nan,
            'y': b.get('y') if b else np.nan
        })
    df = pd.DataFrame(ball_rows).set_index('frame')

    # 2. Interpolate Gaps (Kalman-like smoothing)
    # Fills small gaps to stop the ball from disappearing
    df['x'] = df['x'].interpolate(method='linear', limit=MAX_INTERPOLATION_GAP)
    df['y'] = df['y'].interpolate(method='linear', limit=MAX_INTERPOLATION_GAP)
    
    # 3. Apply Rolling Average (Physical Smoothing)
    # Reduces jitter on frames where the ball is visible
    df['x'] = df['x'].rolling(window=ROLLING_AVG_WINDOW, center=True, min_periods=1).mean()
    df['y'] = df['y'].rolling(window=ROLLING_AVG_WINDOW, center=True, min_periods=1).mean()
    
    df = df.fillna(np.nan).replace([np.nan], [None]) 

    # 4. Inject back into JSON structure
    for i in range(len(frames)):
        if i in df.index:
            x_val = df.loc[i, 'x']
            y_val = df.loc[i, 'y']
            
            if x_val is not None and y_val is not None:
                frames[i]['ball'] = {
                    'x': x_val,
                    'y': y_val,
                    'stable': True 
                }
            else:
                frames[i]['ball'] = None 

    # 5. Save Overwritten Tracks
    Path(OUTPUT_FILE).write_text(json.dumps(tracks_data))
    print("✅ Ball stabilization and smoothing complete.")

if __name__ == '__main__':
    smooth_ball_trajectory()
