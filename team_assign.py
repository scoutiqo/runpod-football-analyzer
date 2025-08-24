# team_assign.py
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

class TeamAssigner:
    """
    Online jersey color clustering.
    - For each player track_id, we maintain an EMA of Lab color from a torso patch.
    - Periodically, we KMeans all track color vectors into 2 clusters.
    - We map the larger cluster -> 'home', smaller -> 'away' (stable and simple).
    """
    def __init__(self, ema_alpha=0.15, min_tracks_for_cluster=6, cluster_every=15):
        self.ema_alpha = ema_alpha
        self.min_tracks_for_cluster = min_tracks_for_cluster
        self.cluster_every = cluster_every

        self.track_color = {}         # id -> np.array([L,a,b])
        self.track_seen = defaultdict(int)
        self.labels = {}              # id -> 'home'|'away'
        self._frame_count = 0
        self._last_fit_frame = -999

    @staticmethod
    def _torso_patch(frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w-1, int(x2)); y2 = min(h-1, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        # crop upper-middle of the box (torso-ish region)
        bw = x2 - x1; bh = y2 - y1
        tx1 = x1 + int(0.25 * bw); tx2 = x1 + int(0.75 * bw)
        ty1 = y1 + int(0.15 * bh); ty2 = y1 + int(0.55 * bh)
        tx1 = max(x1, tx1); tx2 = min(x2, tx2)
        ty1 = max(y1, ty1); ty2 = min(y2, ty2)
        if tx2 <= tx1 or ty2 <= ty1:
            return None
        return frame[ty1:ty2, tx1:tx2, :]

    def observe(self, frame_bgr, players_xyxy_id):
        """
        players_xyxy_id: list of dicts {id, x1,y1,x2,y2}
        Updates EMA Lab color per track id.
        Optionally triggers reclustering every N frames.
        """
        self._frame_count += 1
        for p in players_xyxy_id:
            pid = p["id"]
            patch = self._torso_patch(frame_bgr, p["x1"], p["y1"], p["x2"], p["y2"])
            if patch is None or patch.size == 0:
                continue
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
            # robust mean: clip extremes
            flat = lab.reshape(-1, 3)
            q = np.quantile(flat, [0.1, 0.9], axis=0)
            mask = np.all((flat >= q[0]) & (flat <= q[1]), axis=1)
            vec = flat[mask].mean(axis=0) if mask.any() else flat.mean(axis=0)
            if pid not in self.track_color:
                self.track_color[pid] = vec
            else:
                self.track_color[pid] = (1.0 - self.ema_alpha) * self.track_color[pid] + self.ema_alpha * vec
            self.track_seen[pid] += 1

        # recluster?
        if (self._frame_count - self._last_fit_frame) >= self.cluster_every:
            self._fit_clusters()
            self._last_fit_frame = self._frame_count

    def _fit_clusters(self):
        if len(self.track_color) < self.min_tracks_for_cluster:
            return
        X = np.stack([v for v in self.track_color.values()], axis=0)
        ids = list(self.track_color.keys())
        try:
            km = KMeans(n_clusters=2, n_init=10, random_state=0)
            labels = km.fit_predict(X)
        except Exception:
            return
        # Assign clusters to 'home'/'away' by cluster size (home = larger cluster)
        counts = np.bincount(labels)
        if len(counts) < 2:
            return
        home_label = int(np.argmax(counts))
        for i, pid in enumerate(ids):
            self.labels[pid] = 'home' if labels[i] == home_label else 'away'

    def get_team(self, track_id):
        return self.labels.get(track_id, None)
