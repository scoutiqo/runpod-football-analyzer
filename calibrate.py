import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

# ------------------------------
# Config
# ------------------------------

PITCH_W_M = 105.0
PITCH_H_M = 68.0

# HSV ranges for "green" pitch mask (tweak per league/lighting)
HSV_LOW  = (25, 20, 20)
HSV_HIGH = (85, 255, 255)

SHOTCUT_HIST_BINS = 32
SHOTCUT_THRESH = 0.45   # histogram correlation drop (0..1); lower → more sensitive
MIN_QUAD_AREA_FRAC = 0.08  # quad must occupy at least 15% of frame area
EMA_H_ALPHA = 0.15      # exponential smoothing for homography within a shot


# ------------------------------
# Utility
# ------------------------------

def _order_quad(pts: np.ndarray) -> np.ndarray:
    """
    pts: (4,2) unordered. Return in TL, TR, BR, BL order in image space.
    """
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

def _template_corners() -> np.ndarray:
    """
    Return pitch template corners (meters) in TL, TR, BR, BL order.
    We place (0,0) at top-left in template space for homography fit.
    """
    W, H = PITCH_W_M, PITCH_H_M
    # TL, TR, BR, BL
    return np.array([[0, 0],
                     [W, 0],
                     [W, H],
                     [0, H]], dtype=np.float32)

def _field_mask_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    low = np.array(HSV_LOW, dtype=np.uint8)
    high = np.array(HSV_HIGH, dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    # cleanup
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return mask

def _largest_quad_from_mask(mask: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    From green mask, find the largest contour and approximate a quad.
    Returns 4 points (float32) or None.
    """
    h, w = frame_shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < MIN_QUAD_AREA_FRAC * (w * h):
        return None

    # Approx polygon
    eps = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)

    if approx.shape[0] < 4:
        # fallback to min area rect if not enough vertices
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)  # 4x2
        quad = box.astype(np.float32)
    else:
        # get 4 extreme points via convex hull and then simplify
        hull = cv2.convexHull(approx).reshape(-1, 2)
        if hull.shape[0] > 4:
            # take the 4 farthest corners by rotating calipers-ish heuristic
            # compute min area rectangle on hull
            rect = cv2.minAreaRect(hull.astype(np.float32))
            quad = cv2.boxPoints(rect).astype(np.float32)
        else:
            quad = hull.astype(np.float32)

    if quad.shape[0] != 4:
        return None

    return _order_quad(quad)

def estimate_homography(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Estimate H: image (px) -> template (meters).
    Returns 3x3 homography or None.
    """
    h, w = frame_bgr.shape[:2]
    mask = _field_mask_hsv(frame_bgr)
    quad = _largest_quad_from_mask(mask, (h, w))
    if quad is None:
        return None

    src = quad  # TL, TR, BR, BL (px)
    dst = _template_corners()  # TL, TR, BR, BL (meters)

    H, status = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        return None
    return H

def smooth_homography(prev_H: Optional[np.ndarray], H: np.ndarray, alpha: float = EMA_H_ALPHA) -> np.ndarray:
    if prev_H is None:
        return H
    # normalize both (scale ambiguity)
    Hn = H / (H[2, 2] + 1e-9)
    Pn = prev_H / (prev_H[2, 2] + 1e-9)
    S = (1.0 - alpha) * Pn + alpha * Hn
    return S

def project_px_to_m(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """
    pts_xy: (N,2) image pixel coords.
    Returns (N,2) in meters (template coords).
    """
    pts = np.asarray(pts_xy, dtype=np.float32)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)  # N,3
    p = (H @ pts_h.T).T  # N,3
    p = p[:, :2] / np.clip(p[:, 2:3], 1e-6, None)
    return p

def histogram_cut_metric(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Return L2-normalized concatenated HSV hist as a vector.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [SHOTCUT_HIST_BINS], [0, 256])
        hist.append(h.reshape(-1))
    v = np.concatenate(hist, axis=0).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v

def detect_shotcuts(video_path: str, step: int = 15) -> List[int]:
    """
    Scan the video to find shot-change indices (frame numbers).
    step: sample every N frames for speed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [0]  # at least one segment

    prev = None
    cuts = [0]
    fno = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fno % step == 0:
            v = histogram_cut_metric(frame)
            if prev is not None:
                sim = float(np.dot(prev, v))
                # correlation-ish in [0,1]; low means change
                if sim < (1.0 - SHOTCUT_THRESH):
                    cuts.append(fno)
            prev = v
        fno += 1
    cap.release()
    # unique & sorted
    cuts = sorted(set(cuts))
    return cuts

# ------------------------------
# Public API to use in pipeline
# ------------------------------

@dataclass
class CalibResult:
    H_by_frame: Dict[int, np.ndarray]
    shot_cuts: List[int]
    pitch_m: Tuple[float, float] = (PITCH_W_M, PITCH_H_M)

def calibrate_video(video_path: str, sample_every: int = 10) -> CalibResult:
    """
    Estimate homography per shot and propagate/smooth across frames.
    We evaluate H at sparse frames (sample_every) and hold/smooth between.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cuts = detect_shotcuts(video_path, step=15)
    cuts = sorted([c for c in cuts if 0 <= c < total_frames])
    if cuts and cuts[0] != 0:
        cuts.insert(0, 0)

    # homography per shot (use first decent frame after cut)
    H_by_frame: Dict[int, np.ndarray] = {}
    last_H = None

    for idx_cut, cut_fno in enumerate(cuts):
        # seek to first frame after cut
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cut_fno))
        H_this = None
        # probe a few frames to get a good estimate
        for _ in range(12):
            ok, frame = cap.read()
            if not ok:
                break
            H_est = estimate_homography(frame)
            if H_est is not None:
                # smooth with previous shot's H to avoid jumps
                H_this = smooth_homography(last_H, H_est, alpha=EMA_H_ALPHA)
                break
        if H_this is None:
            # fallback to previous H
            H_this = last_H
        # store "anchor" for this frame
        if H_this is not None:
            H_by_frame[cut_fno] = H_this
            last_H = H_this

    cap.release()

    # Fill-in between cuts with the last known H
    # Optionally, you could re-sample every N frames and re-estimate.
    # Here we keep per-shot constant H (already smoothed at boundary).
    return CalibResult(H_by_frame=H_by_frame, shot_cuts=cuts, pitch_m=(PITCH_W_M, PITCH_H_M))

def select_H_for_frame(fno: int, H_by_frame: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    """Pick the nearest past anchor H for a given frame number."""
    anchors = sorted(H_by_frame.keys())
    if not anchors:
        return None
    idx = max([a for a in anchors if a <= fno], default=anchors[0])
    return H_by_frame.get(idx)

def map_tracks_px_to_m(tracks_raw: List[Dict], fps: float, H_by_frame: Dict[int, np.ndarray]) -> List[Dict]:
    """
    tracks_raw: list of {t, type, id, x_px, y_px, ...}
    fps: frames per second of the original video
    H_by_frame: dict frame_index -> H (px->m)
    """
    out = []
    for r in tracks_raw:
        t = float(r.get("t", 0.0))
        fno = int(round(t * fps))
        H = select_H_for_frame(fno, H_by_frame)
        if H is None or ('x_px' not in r) or ('y_px' not in r):
            # pass-through if missing
            out.append(dict(r))
            continue
        px = np.array([[float(r['x_px']), float(r['y_px'])]], dtype=np.float32)
        pm = project_px_to_m(H, px)[0]
        rr = dict(r)
        rr['x_m'] = float(pm[0])
        rr['y_m'] = float(pm[1])
        out.append(rr)
    return out


# ------------------------------
# CLI helper (optional)
# ------------------------------

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _dump_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def process_segment(video_path: str, tracks_raw_path: str, fps: float, out_path: str):
    """
    Calibrate a single (possibly long) video + map tracks to meters.
    """
    calib = calibrate_video(video_path)
    tracks_raw = _load_json(tracks_raw_path)
    tracks_list = tracks_raw.get("tracks", tracks_raw if isinstance(tracks_raw, list) else [])

    tracks_m = map_tracks_px_to_m(tracks_list, fps, calib.H_by_frame)
    out = {"video": os.path.basename(video_path),
           "pitch": {"length_m": PITCH_W_M, "width_m": PITCH_H_M},
           "tracks": tracks_m}
    _dump_json(out, out_path)
    print(f"[calibrate] wrote {out_path} with {len(tracks_m)} items")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks_raw", required=True, help="tracks with x_px/y_px")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--out", required=True, help="output tracks.json (meters)")
    args = ap.parse_args()
    process_segment(args.video, args.tracks_raw, args.fps, args.out)
