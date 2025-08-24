import numpy as np

def estimate_homography(_first_frame_bgr):
    """
    TODO: implement pitch line/keypoint detector -> RANSAC homography.
    For now return None and we work in pixels (frontend still works).
    """
    return None

def image_to_field(H, x_px, y_px):
    if H is None:
        return None
    p = np.array([x_px, y_px, 1.0], dtype=float)
    q = H @ p
    q /= q[2]
    return float(q[0]), float(q[1])
