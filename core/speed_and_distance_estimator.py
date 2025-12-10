import cv2
import numpy as np

class SpeedAndDistanceEstimator:
    def __init__(self):
        self.frame_window = 5 # Smooth over 5 frames (0.2s)
        self.frame_rate = 25.0

    def measure_distance(self, p1, p2):
        # Euclidean distance
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def add_speed_and_distance_to_tracks(self, tracks):
        # Organise data by Player ID for time-series analysis
        player_stats = {}
        frames = tracks.get('frames', [])
        
        if not frames: return tracks

        # 1. Extract positions
        for i, f in enumerate(frames):
            for p in f['players']:
                pid = str(p['id'])
                if pid not in player_stats: player_stats[pid] = []
                
                # TRUST THE PIPELINE: Use x_m/y_m if they exist and are valid (>0)
                # This ensures we use the 3D depth from MonoLoco
                xm = p.get('x_m', -1)
                ym = p.get('y_m', -1)
                
                if xm != -1 and ym != -1:
                    player_stats[pid].append({
                        'frame': i,
                        'pos': [xm, ym]
                    })

        # 2. Calculate Metrics
        for pid, points in player_stats.items():
            total_dist = 0.0
            
            for k in range(len(points)):
                if k < self.frame_window: continue
                
                curr = points[k]
                prev = points[k - self.frame_window]
                
                # Ensure frames are contiguous (no tracking loss in between)
                if (curr['frame'] - prev['frame']) == self.frame_window:
                    dist_m = self.measure_distance(prev['pos'], curr['pos'])
                    time_s = self.frame_window / self.frame_rate
                    
                    # Speed in km/h
                    speed_kmh = (dist_m / time_s) * 3.6
                    
                    # Sanity Cap (40km/h) to filter noise
                    if speed_kmh > 40: speed_kmh = 0
                    
                    # Inject back into the original JSON structure
                    frame_idx = curr['frame']
                    for p in frames[frame_idx]['players']:
                        if str(p['id']) == pid:
                            p['speed'] = round(float(speed_kmh), 2) # Float cast for JSON safety

                # Total Distance
                if k > 0:
                    prev_step = points[k-1]
                    if (curr['frame'] - prev_step['frame']) == 1:
                        dist = self.measure_distance(prev_step['pos'], curr['pos'])
                        if dist < 2.0: # Ignore teleportation jumps > 2m per frame
                            total_dist += dist
                            
                    frame_idx = curr['frame']
                    for p in frames[frame_idx]['players']:
                         if str(p['id']) == pid:
                             p['total_distance'] = round(float(total_dist / 1000), 3) # KM

        return tracks
