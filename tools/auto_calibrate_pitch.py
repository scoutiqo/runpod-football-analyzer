import argparse
from pathlib import Path
import json

import cv2
import numpy as np


def detect_pitch_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # broad green range; tweak if needed
    lower = np.array([25, 30, 30])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def cluster_lines(lines):
    thetas = []
    good_lines = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = np.arctan2(dy, dx)
        thetas.append(angle)
        good_lines.append((x1, y1, x2, y2))
    thetas = np.array(thetas, dtype=np.float32)
    if len(thetas) < 4:
        return [], []
    # map angles to [0, pi)
    thetas = (thetas + np.pi) % np.pi
    data = thetas.reshape(-1, 1)
    # k=2 clusters: roughly vertical vs horizontal
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    ret, labels, centers = cv2.kmeans(
        data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    cluster1 = []
    cluster2 = []
    for (x1, y1, x2, y2), lab in zip(good_lines, labels.flatten()):
        if lab == 0:
            cluster1.append((x1, y1, x2, y2))
        else:
            cluster2.append((x1, y1, x2, y2))
    return cluster1, cluster2


def fit_lines(lines):
    # fit line in ax + by + c = 0 form via least squares
    fitted = []
    for (x1, y1, x2, y2) in lines:
        pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        a = -vy
        b = vx
        c = -(a * x0 + b * y0)
        fitted.append((float(a), float(b), float(c)))
    return fitted


def intersections(lines_a, lines_b, width, height):
    pts = []
    for (a1, b1, c1) in lines_a:
        for (a2, b2, c2) in lines_b:
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-6:
                continue
            x = (b1 * c2 - b2 * c1) / det
            y = (c1 * a2 - c2 * a1) / det
            if 0 <= x <= width and 0 <= y <= height:
                pts.append((x, y))
    return pts


def order_corners(corners):
    pts = np.array(corners, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def auto_calibrate(frame_path, pitch_w=105.0, pitch_h=68.0):
    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {frame_path}")
    h, w = img.shape[:2]

    mask = detect_pitch_mask(img)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=min(w, h) * 0.25,
        maxLineGap=20,
    )
    if lines is None or len(lines) < 4:
        raise RuntimeError("Not enough lines detected for calibration")

    cluster1, cluster2 = cluster_lines(lines)
    if len(cluster1) < 2 or len(cluster2) < 2:
        raise RuntimeError("Could not cluster lines into two groups")

    lines_a = fit_lines(cluster1)
    lines_b = fit_lines(cluster2)

    pts = intersections(lines_a, lines_b, w, h)
    if len(pts) < 4:
        raise RuntimeError("Not enough intersections to estimate corners")

    pts_np = np.array(pts, dtype=np.float32)
    corners = order_corners(pts_np)

    dst = np.array(
        [
            [0.0, 0.0],
            [pitch_w, 0.0],
            [pitch_w, pitch_h],
            [0.0, pitch_h],
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(corners, dst, method=0)
    if H is None:
        raise RuntimeError("findHomography failed")

    return H, corners, (w, h)


def main():
    parser = argparse.ArgumentParser(description="Automatic pitch homography calibration")
    parser.add_argument("--frame", required=True, help="calibration_frame.jpg")
    parser.add_argument("--out", required=True, help="homography_auto.json")
    parser.add_argument("--pitch-w", type=float, default=105.0)
    parser.add_argument("--pitch-h", type=float, default=68.0)
    args = parser.parse_args()

    frame_path = Path(args.frame)
    H, corners, (w, h) = auto_calibrate(frame_path, args.pitch_w, args.pitch_h)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "frame_image": str(frame_path),
        "image_size": [int(w), int(h)],
        "pitch_m": [float(args.pitch_w), float(args.pitch_h)],
        "corners_px": corners.tolist(),  # TL, TR, BR, BL
        "matrix": H.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved automatic homography to {out_path}")


if __name__ == "__main__":
    main()
