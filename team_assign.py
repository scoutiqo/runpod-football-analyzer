import cv2
import numpy as np
from collections import defaultdict, deque
from sklearn.cluster import KMeans

class TeamAssigner:
    """
    Online jersey color clustering with referee/unknown handling and temporal voting.

    - Extracts a torso patch per track bbox and computes its LAB mean (EMA smoothed).
    - KMeans into 2 clusters (home/away). If many dark samples exist, try 3 clusters (referee).
    - Marks outliers far from all centers as 'unknown' or 'ref' (very dark).
    - Temporal voting over recent frames to stabilize labels.
    """
    def __init__(self,
                 ema_alpha=0.15,
                 min_tracks_for_cluster=6,
                 cluster_every=15,
                 outlier_tau=22.0,
                 vote_window=15):
        self.ema_alpha = float(ema_alpha)
        self.min_tracks_for_cluster = int(min_tracks_for_cluster)
        self.cluster_every = int(cluster_every)
        self.outlier_tau = float(outlier_tau)
        self.vote_window = int(vote_window)

        self.track_color = {}                 # id -> np.array([L,a,b])
        self.track_seen = defaultdict(int)    # id -> count
        self.labels = {}                      # id -> 'home'|'away'|'ref'|'unknown'
        self._label_history = defaultdict(lambda: deque(maxlen=self.vote_window))

        self._frame_count = 0
        self._last_fit_frame = -999

        self.kmeans_centers = None            # (k,3) LAB
        self.home_cluster = 0                 # which kmeans label = 'home'

    @staticmethod
    def _torso_patch(frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
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
        Updates EMA LAB color per track id and reclusters periodically.
        """
        self._frame_count += 1
        for p in players_xyxy_id:
            pid = p["id"]
            patch = self._torso_patch(frame_bgr, p["x1"], p["y1"], p["x2"], p["y2"])
            if patch is None or patch.size == 0:
                continue
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
            flat = lab.reshape(-1, 3).astype(np.float32)
            # robust mean: clip extremes
            q = np.quantile(flat, [0.1, 0.9], axis=0)
            mask = np.all((flat >= q[0]) & (flat <= q[1]), axis=1)
            vec = (flat[mask].mean(axis=0) if mask.any() else flat.mean(axis=0)).astype(np.float32)
            if pid not in self.track_color:
                self.track_color[pid] = vec
            else:
                self.track_color[pid] = (1.0 - self.ema_alpha) * self.track_color[pid] + self.ema_alpha * vec
            self.track_seen[pid] += 1

        if (self._frame_count - self._last_fit_frame) >= self.cluster_every:
            self._fit_clusters()
            self._last_fit_frame = self._frame_count

    def _fit_clusters(self):
        if len(self.track_color) < self.min_tracks_for_cluster:
            return
        ids = list(self.track_color.keys())
        X = np.stack([self.track_color[i] for i in ids], axis=0).astype(np.float32)

        # Start with 2 clusters (home/away)
        try:
            km = KMeans(n_clusters=2, n_init=10, random_state=0)
            labels = km.fit_predict(X)
            centers = km.cluster_centers_.astype(np.float32)
        except Exception:
            return

        # If many very dark samples, try 3 clusters to capture referees
        # NOTE: OpenCV LAB L channel is 0..255 (not 0..100).
        if (X[:, 0] < 35).mean() > 0.15:
            try:
                km3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
                labels = km3.labels_
                centers = km3.cluster_centers_.astype(np.float32)
            except Exception:
                pass

        counts = np.bincount(labels)
        if counts.size < 2:
            return

        self.kmeans_centers = centers
        home_label = int(np.argmax(counts))
        self.home_cluster = home_label

        # initial mapping home/away by cluster size
        tmp = {}
        for i, pid in enumerate(ids):
            lab = labels[i]
            tmp[pid] = 'home' if lab == home_label else 'away'

        # Outliers / referee / unknown handling
        if self.kmeans_centers is not None:
            for pid in ids:
                v = self.track_color.get(pid)
                if v is None:
                    continue
                d = np.min(np.linalg.norm(self.kmeans_centers - v, axis=1))
                if d > self.outlier_tau:
                    # very dark => referee; else unknown
                    if v[0] < 35:  # L channel threshold (0..255)
                        tmp[pid] = 'ref'
                    else:
                        tmp[pid] = 'unknown'

        # Temporal voting to stabilize label output
        for pid, lbl in tmp.items():
            hist = self._label_history[pid]
            hist.append(lbl)
            vals, cnts = np.unique(list(hist), return_counts=True)
            self.labels[pid] = str(vals[int(np.argmax(cnts))])

    def get_team(self, track_id):
        return self.labels.get(track_id, None)
