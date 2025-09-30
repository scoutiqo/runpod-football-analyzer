import numpy as np
import cv2
from sklearn.cluster import KMeans

def _upper_body_crop(frame, xyxy):
    x1,y1,x2,y2 = map(int, xyxy)
    x1,x2 = max(0,x1), max(0,x2)
    y1,y2 = max(0,y1), max(0,y2)
    if x2<=x1 or y2<=y1: 
        return None
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    h = patch.shape[0]
    return patch[: max(2, h//2), :]

def _feat_from_patch(patch):
    # robust to lighting: HSV mean + 16-bin H histogram
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean = hsv.reshape(-1,3).mean(axis=0)            # (H,S,V)
    hist = cv2.calcHist([hsv],[0],None,[16],[0,180]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    return np.concatenate([mean, hist]).astype(np.float32)

class TeamAssigner:
    def __init__(self, max_samples=300):
        self.features = []
        self.km = None
        self.max_samples = max_samples
        self.colors_hex = ["#00BFFF", "#FF6347"]  # blue vs tomato

    def add_samples(self, frame, xyxys):
        for b in xyxys:
            patch = _upper_body_crop(frame, b)
            if patch is None: 
                continue
            f = _feat_from_patch(patch)
            self.features.append(f)
            if len(self.features) >= self.max_samples:
                break

    def fit(self):
        if len(self.features) < 20:
            return False
        X = np.stack(self.features, axis=0)
        self.km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0).fit(X)
        return True

    def predict_ids(self, frame, xyxys):
        if self.km is None or len(xyxys)==0:
            return None
        feats = []
        ok = []
        for b in xyxys:
            p = _upper_body_crop(frame, b)
            if p is None:
                ok.append(False)
                feats.append(np.zeros(19, dtype=np.float32))
            else:
                ok.append(True)
                feats.append(_feat_from_patch(p))
        feats = np.stack(feats, axis=0)
        labels = self.km.predict(feats)
        return labels  # 0/1 per player det
