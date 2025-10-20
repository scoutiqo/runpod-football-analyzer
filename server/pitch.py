# server/pitch.py
import cv2, numpy as np

def build_pitch_mask(frame_bgr, exclude_rect=None):
    """
    Returns a binary mask (uint8) where 255 = in-pitch, 0 = not pitch.
    exclude_rect: optional (x1,y1,x2,y2) to blank out stands/crowd band.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Broad green range; tweak if needed for your feed
    lower = np.array([30, 35, 35], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up & keep the largest green blob (the pitch)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)

    # Keep largest connected component only
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + np.argmax(areas)
        mask = np.where(labels == largest_idx, 255, 0).astype("uint8")

    if exclude_rect:
        x1,y1,x2,y2 = exclude_rect
        x1=max(0,x1); y1=max(0,y1); x2=min(w-1,x2); y2=min(h-1,y2)
        mask[y1:y2, x1:x2] = 0

    return mask

def box_kept_by_mask(box, mask, feet_ratio=0.95, min_overlap=0.3):
    """
    Keep detection if:
      - the FEET point (x_mid, y2*feet_ratio) is on the mask, AND
      - IoA (overlap / box area) with the mask >= min_overlap.
    """
    x1,y1,x2,y2 = map(int, box)
    h, w = mask.shape[:2]
    x1=max(0,x1); y1=max(0,y1); x2=min(w-1,x2); y2=min(h-1,y2)
    if x2<=x1 or y2<=y1: return False

    # Feet test
    xf = int((x1+x2)/2); yf = int(y1 + feet_ratio*(y2-y1))
    if mask[yf, xf] == 0:
        return False

    # IoA test
    roi = mask[y1:y2, x1:x2]
    pitch_pixels = int((roi == 255).sum())
    area = max(1, (x2-x1) * (y2-y1))
    return (pitch_pixels / area) >= min_overlap
