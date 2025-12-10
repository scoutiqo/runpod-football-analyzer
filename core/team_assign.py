import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, Counter

class TeamAssigner:
    def __init__(self):
        self.player_colors = defaultdict(list)
        self.final_labels = {} 

    def _torso_patch(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1: return None
        
        bw, bh = x2-x1, y2-y1
        # Crop center torso (avoiding shorts/grass background)
        tx1 = x1 + int(0.3 * bw)
        tx2 = x1 + int(0.7 * bw)
        ty1 = y1 + int(0.2 * bh)
        ty2 = y1 + int(0.5 * bh)
        return frame[ty1:ty2, tx1:tx2]

    def observe(self, frame_bgr, players):
        for p in players:
            pid = p["id"]
            patch = self._torso_patch(frame_bgr, p["x1"], p["y1"], p["x2"], p["y2"])
            if patch is None or patch.size == 0: continue
            
            # Use LAB color space (better for human perception)
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
            mean_color = np.median(lab.reshape(-1, 3), axis=0)
            
            if len(self.player_colors[pid]) < 30:
                self.player_colors[pid].append(mean_color)

    def fit_global(self):
        if not self.player_colors: return
        
        # 1. Get average color per player
        player_avgs = []
        pids = []
        for pid, colors in self.player_colors.items():
            avg = np.mean(colors, axis=0)
            player_avgs.append(avg)
            pids.append(pid)
            
        X = np.array(player_avgs)
        
        # 2. K-Means with K=2 (Primary Teams)
        kmeans = KMeans(n_clusters=2, n_init=20, random_state=42).fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        # 3. OUTLIER REJECTION (The "Not a Player" Filter)
        # Calculate distance of each player to their assigned center
        distances = np.linalg.norm(X - centers[labels], axis=1)
        
        # Dynamic Threshold: Reject if distance is > 1.5x the median distance
        # This filters out Refs (Yellow/Black) and Coaches (Dark coats)
        median_dist = np.median(distances)
        threshold = median_dist * 2.0
        
        self.final_labels = {}
        for i, pid in enumerate(pids):
            if distances[i] > threshold:
                self.final_labels[pid] = 'unknown' # Ref/Coach
            else:
                # Map 0/1 to A/B
                self.final_labels[pid] = str(labels[i])

    def get_team(self, track_id):
        return self.final_labels.get(track_id, 'unknown')
