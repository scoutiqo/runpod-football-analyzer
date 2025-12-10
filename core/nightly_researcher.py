import json
import numpy as np
import glob
import math
from pathlib import Path
from scipy.interpolate import interp1d
import os

# CONFIG
MASTER_DIR = "runs/json"
REPORT_FILE = "nightly_report.json"

class NightlyResearcher:
    def __init__(self):
        self.report = {
            "ball_frames_recovered": 0,
            "shots_discovered": 0,
            "camera_cuts_detected": 0,
            "fixed_jobs": []
        }

    def interpolate_ball(self, frames):
        """
        Fixes flickering ball tracking using linear interpolation.
        """
        ball_positions = []
        frames_indices = []
        recovered = 0
        
        # 1. Extract existing ball points
        for i, f in enumerate(frames):
            ball = f.get('ball')
            # Handle corrupted ball data (sometimes list, sometimes dict)
            if ball and isinstance(ball, dict) and 'x' in ball:
                ball_positions.append([ball['x'], ball['y']])
                frames_indices.append(i)
        
        if len(ball_positions) < 2: return frames, 0
        
        # 2. Create Math Model
        ball_positions = np.array(ball_positions)
        try:
            f_x = interp1d(frames_indices, ball_positions[:, 0], kind='linear', fill_value="extrapolate")
            f_y = interp1d(frames_indices, ball_positions[:, 1], kind='linear', fill_value="extrapolate")
            
            # 3. Fill Gaps
            for i in range(len(frames)):
                if i < frames_indices[0] or i > frames_indices[-1]: continue
                
                if not frames[i].get('ball'):
                    new_x = float(f_x(i))
                    new_y = float(f_y(i))
                    # Sanity Check: Don't interpolate if it goes off screen
                    if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                        frames[i]['ball'] = {"x": new_x, "y": new_y, "interpolated": True}
                        recovered += 1
        except Exception:
            pass 
                
        return frames, recovered

    def discover_shots(self, frames):
        """
        Stricter Shot Logic: Must hit the GOAL MOUTH, not just the end line.
        """
        shots_found = 0
        potential_shots = []
        
        for i in range(5, len(frames)-5):
            f_curr = frames[i]
            f_next = frames[i+5] # Check 0.2s later
            
            b1 = f_curr.get('ball')
            b2 = f_next.get('ball')
            
            if not b1 or not b2: continue
            if not isinstance(b1, dict) or not isinstance(b2, dict): continue
            
            # Calc Velocity
            dx = b2.get('x', 0) - b1.get('x', 0)
            dy = b2.get('y', 0) - b1.get('y', 0)
            dist = math.hypot(dx, dy)
            
            # STRICT SHOT HEURISTIC:
            # 1. Very fast (dist > 0.15 in 5 frames)
            # 2. Ends at Goal Line (x < 0.05 or x > 0.95)
            # 3. AND is Central (Goal Mouth) (0.42 < y < 0.58) <--- NEW CONSTRAINT
            
            dest_x = b2.get('x', 0.5)
            dest_y = b2.get('y', 0.5)
            
            is_fast = dist > 0.15
            is_end_line = (dest_x < 0.05 or dest_x > 0.95)
            is_goal_mouth = (0.42 < dest_y < 0.58)
            
            if is_fast and is_end_line and is_goal_mouth:
                # Debounce: Don't count the same shot 5 times in a row
                if not potential_shots or (i - potential_shots[-1]) > 25:
                    shots_found += 1
                    potential_shots.append(i)
                    
        return potential_shots, shots_found

    def run(self):
        print("🕵️ Nightly Researcher Active (Strict Mode)...")
        
        track_files = glob.glob(f"{MASTER_DIR}/tracks_*.json")
        
        for tf in track_files:
            if "improved_" in tf: continue 
            
            job_id = Path(tf).stem.replace("tracks_", "")
            print(f"   🔬 Analyzing Job: {job_id}...")
            
            try:
                raw_data = json.loads(Path(tf).read_text())
                
                # FIX: Handle List vs Dict structure
                if isinstance(raw_data, list):
                    frames = raw_data # Old format
                    final_structure = {"fps": 25, "frames": frames} # Convert to new
                else:
                    frames = raw_data.get('frames', [])
                    final_structure = raw_data
                
                if not frames: continue
                
                # 1. Fix Tracking
                frames, recovered = self.interpolate_ball(frames)
                self.report["ball_frames_recovered"] += recovered
                
                # 2. Discover Shots (Strict)
                shot_frames, shot_count = self.discover_shots(frames)
                self.report["shots_discovered"] += shot_count
                
                # 3. Save "Improved" Tracks
                improved_path = Path(tf).parent / f"improved_tracks_{job_id}.json"
                
                for idx in shot_frames:
                    frames[idx]['suggested_event'] = "shot"
                    
                final_structure['frames'] = frames
                improved_path.write_text(json.dumps(final_structure))
                
                self.report["fixed_jobs"].append(job_id)
                print(f"      -> Fixed {recovered} ball frames. Found {shot_count} shots.")
                
            except Exception as e:
                print(f"      ⚠️ Failed to process {job_id}: {e}")

        Path(REPORT_FILE).write_text(json.dumps(self.report, indent=2))
        print("\n✅ Research Complete.")
        print(json.dumps(self.report, indent=2))

if __name__ == "__main__":
    agent = NightlyResearcher()
    agent.run()
