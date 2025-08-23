# calibrate.py
import cv2, numpy as np

def estimate_homography(first_frame_bgr):
    """
    Try to find pitch lines + keypoints and solve H (image->field).
    Return H (3x3) or None. Field is 105x68 (meters).
    """
    # Minimal placeholder: return None to run in pixel mode initially.
    # TODO:
    #  - Detect sidelines/box lines via Hough + template
    #  - Match to canonical field points, RANSAC homography
    return None

def image_to_field(H, x_px, y_px, pitch_size=(105,68), img_shape=None):
    if H is None:  # px mode fallback
        return None
    p = np.array([x_px, y_px, 1.0])
    q = H @ p
    q /= q[2]
    return float(q[0]), float(q[1])
