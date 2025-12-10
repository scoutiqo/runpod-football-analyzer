import cv2, numpy as np

class PitchFilter:
    def __init__(self, green_low=(35,40,40), green_high=(85,255,255), on_pitch_thresh=0.6):
        self.green_low  = np.array(green_low, dtype=np.uint8)
        self.green_high = np.array(green_high, dtype=np.uint8)
        self.on_pitch_thresh = on_pitch_thresh
        self.last_team_colors = None  # optional persistence

    def _pitch_mask(self, frame):
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_low, self.green_high)
        # optional: open/close to denoise
        k = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _on_pitch(self, box, mask):
        x1,y1,x2,y2 = [int(v) for v in box]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(mask.shape[1], x2); y2 = min(mask.shape[0], y2)
        if x2 <= x1 or y2 <= y1: return False
        crop = mask[y1:y2, x1:x2]
        return (crop > 0).mean() >= self.on_pitch_thresh

    def _cluster_jerseys(self, frame, boxes, k=3):
        # Compute mean HSV per box, then kmeans; return cluster labels and cluster sizes
        if not boxes: return [], {}
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        feats = []
        for (x1,y1,x2,y2) in boxes:
            x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
            x1=max(0,x1); y1=max(0,y1); x2=min(hsv.shape[1],x2); y2=min(hsv.shape[0],y2)
            if x2<=x1 or y2<=y1:
                feats.append([0,0,0]); continue
            patch = hsv[y1:y2, x1:x2]
            mean = patch.reshape(-1,3).mean(axis=0)
            feats.append(mean.tolist())
        data = np.float32(feats)
        # kmeans
        K = min(k, len(boxes))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten().tolist()
        counts = {}
        for lb in labels: counts[lb] = counts.get(lb,0)+1
        return labels, counts

    def filter_detections(self, frame, boxes):
        """
        boxes: list of [x1,y1,x2,y2] (floats/ints)
        returns filtered list of boxes (same format)
        """
        if not boxes: return boxes
        pitch = self._pitch_mask(frame)
        on_pitch_boxes = []
        for b in boxes:
            if self._on_pitch(b, pitch):
                on_pitch_boxes.append(b)
        if len(on_pitch_boxes) <= 2:
            return on_pitch_boxes

        labels, counts = self._cluster_jerseys(frame, on_pitch_boxes, k=3)
        if not labels:
            return on_pitch_boxes

        # Drop smallest cluster (likely ref/liners). Keep top-2 clusters.
        sorted_clusters = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        keep_clusters = {c for c,_ in sorted_clusters[:2]}
        kept = [b for b,lb in zip(on_pitch_boxes, labels) if lb in keep_clusters]
        return kept
