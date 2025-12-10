import cv2
import numpy as np
import json
from sklearn.cluster import KMeans
from collections import defaultdict
from pathlib import Path

OUTPUT_COLORS = "runs/json/team_colors.json"

class TeamAssigner:
    def __init__(self):
        self.player_colors = defaultdict(list)
        self.final_labels = {} 

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        img = frame[y1:y2, x1:x2]
        if img.size == 0: return None

        # Heuristic: Shirt is usually in the upper-middle 
        h_crop, w_crop = img.shape[:2]
        shirt_patch = img[int(h_crop*0.15):int(h_crop*0.6), int(w_crop*0.25):int(w_crop*0.75)]
        
        if shirt_patch.size == 0: return None
        
        # Return average color of the shirt patch
        return np.mean(shirt_patch.reshape(-1, 3), axis=0)

    def observe(self, frame, players):
        for p in players:
            color = self.get_player_color(frame, [p["x1"], p["y1"], p["x2"], p["y2"]])
            if color is not None:
                self.player_colors[p["id"]].append(color)

    def fit_global(self):
        print(f"🤖 Analyzing Team Colors...")
        
        # 1. Condense history to ONE average color per Unique Player ID
        player_vectors = []
        pids = []
        
        for pid, samples in self.player_colors.items():
            if len(samples) < 5: continue # Ignore noise tracks
            avg_color = np.mean(samples, axis=0)
            player_vectors.append(avg_color)
            pids.append(pid)
            
        if len(player_vectors) < 2: return

        data = np.array(player_vectors)
        
        # 2. Attempt K-Means
        try:
            kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
            labels = kmeans.fit_predict(data)
            centers = kmeans.cluster_centers_
        except:
            labels = [0] * len(data)
            centers = [np.mean(data, axis=0), np.mean(data, axis=0)]

        # 3. The "Civil War" Check (Force Split if unbalanced)
        count0 = np.sum(labels == 0)
        count1 = np.sum(labels == 1)
        total = len(labels)
        
        # If split is worse than 80/20, K-Means failed.
        # We fallback to sorting by Hue and cutting the list.
        if count0 < total * 0.2 or count1 < total * 0.2:
            print("   ⚠️ K-Means failed (Unbalanced). Forcing Hue Split...")
            
            # Convert all to Hue
            hues = []
            for bgr in data:
                hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                hues.append(hsv[0]) # Hue channel
            
            # Sort PIDs by Hue
            sorted_indices = np.argsort(hues)
            midpoint = len(hues) // 2
            
            # Force labels based on sorted hue
            new_labels = np.zeros(len(data), dtype=int)
            for i in range(len(data)):
                original_idx = sorted_indices[i]
                if i < midpoint:
                    new_labels[original_idx] = 0
                else:
                    new_labels[original_idx] = 1
            
            labels = new_labels
            
            # Re-calculate centers for frontend display
            c0 = data[labels == 0]
            c1 = data[labels == 1]
            centers[0] = np.mean(c0, axis=0) if len(c0) > 0 else centers[0]
            centers[1] = np.mean(c1, axis=0) if len(c1) > 0 else centers[1]

        # 4. Assign
        self.final_labels = {str(pids[i]): ('A' if l == 0 else 'B') for i, l in enumerate(labels)}
        
        # 5. Save
        def to_hex(c): return "#{:02x}{:02x}{:02x}".format(int(c[2]), int(c[1]), int(c[0]))
        
        out = {"A": {"hex": to_hex(centers[0])}, "B": {"hex": to_hex(centers[1])}}
        Path(OUTPUT_COLORS).parent.mkdir(parents=True, exist_ok=True)
        Path(OUTPUT_COLORS).write_text(json.dumps(out))
        
        print(f"   ✅ Teams Assigned: A={np.sum(labels==0)}, B={np.sum(labels==1)}")

    def get_team(self, pid):
        return self.final_labels.get(str(pid), 'unknown')
