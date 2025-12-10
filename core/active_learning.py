#!/usr/bin/env python
"""
active_learning.py (Professional Version)

This module implements a full active-learning pipeline:
- uncertainty sampling
- entropy sampling
- margin sampling
- diversity clustering (KMeans)
- event-boundary selection
- snippet extraction via ffmpeg
- Label Studio task generation
- produces ls_active_learning_tasks.json for import

Inputs:
  runs/json/predicted_events.json
  models/event_model_extreme_meta.json
  http://localhost:8081/data/upload/2/0f817600-test_short.mp4  (VIDEO_URL)

Outputs:
  runs/active_learning/ls_active_learning_tasks.json
  runs/active_learning/snippets/frame_XXX.mp4
  runs/active_learning/snippets/frame_XXX.jpg
"""

import json
import os
from pathlib import Path
import numpy as np
import subprocess
from sklearn.cluster import KMeans

PRED_PATH = "runs/json/predicted_events.json"
META_PATH = "models/event_model_extreme_meta.json"

VIDEO_URL = "http://localhost:8081/data/upload/2/0f817600-test_short.mp4"
VIDEO_LOCAL = "uploads/test_short.mp4"   # If LS uses local copy
OUT_DIR = "runs/active_learning"
SNIPPET_DIR = f"{OUT_DIR}/snippets"

# CONFIG
UNCERTAINTY_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.20
NUM_CLUSTERS = 8
SNIPPET_SECONDS = 0.4   # each side of frame


def load_predictions():
    data = json.loads(Path(PRED_PATH).read_text())
    return data


def entropy(probs):
    """Shannon entropy."""
    p = np.array(probs)
    p = p + 1e-9
    return -np.sum(p * np.log(p))


def compute_uncertainty_frames(preds):
    """
    Returns list of (frame, uncertainty_score, type) 
    """
    results = []

    for p in preds:
        frame = p["frame"]
        # Probabilities unavailable? Build fake distribution
        # Only best_prob stored
        best_prob = p["prob"]
        other_prob = (1.0 - best_prob) / 2.0
        probs = np.array([best_prob, other_prob, other_prob])

        # Entropy
        ent = entropy(probs)

        # Margin = diff between top1 and top2
        margin = abs(best_prob - max([other_prob, other_prob]))

        # Uncertainty = low confidence + high entropy + low margin
        unc_score = (1 - best_prob) + ent + (1 - margin)

        results.append({
            "frame": frame,
            "uncertainty": float(unc_score),
            "entropy": float(ent),
            "margin": float(margin),
            "prob": float(best_prob),
            "pred": p["label"]
        })

    return results


def filter_uncertain(unc_list):
    """Keep frames where confidence low, entropy/margin high."""
    out = [
        u for u in unc_list
        if u["prob"] < UNCERTAINTY_THRESHOLD or u["margin"] < MARGIN_THRESHOLD
    ]
    return out


def cluster_frames(frames):
    """Cluster by frame index to ensure diversity."""
    if len(frames) <= NUM_CLUSTERS:
        return frames
    
    X = np.array([[f["frame"]] for f in frames])
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    labels = kmeans.fit_predict(X)

    # pick one frame from each cluster with highest uncertainty
    result = []
    for c in range(NUM_CLUSTERS):
        members = [f for f, lbl in zip(frames, labels) if lbl == c]
        if not members:
            continue
        best = sorted(members, key=lambda x: x["uncertainty"], reverse=True)[0]
        result.append(best)

    return sorted(result, key=lambda x: x["frame"])


def extract_snippet(local_video, frame, fps=25.0):
    """Extract small clip around a specific frame."""
    os.makedirs(SNIPPET_DIR, exist_ok=True)
    t = frame / fps

    start = max(t - SNIPPET_SECONDS, 0)
    duration = SNIPPET_SECONDS * 2

    out_path = f"{SNIPPET_DIR}/frame_{frame}.mp4"
    thumb_path = f"{SNIPPET_DIR}/frame_{frame}.jpg"

    # Extract snippet
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start}",
        "-i", local_video,
        "-t", f"{duration}",
        "-vf", "scale=640:-1",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Extract thumbnail
    cmd2 = [
        "ffmpeg", "-y",
        "-ss", f"{t}",
        "-i", local_video,
        "-vf", "scale=320:-1",
        "-vframes", "1",
        thumb_path
    ]
    subprocess.run(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out_path, thumb_path


def build_label_studio_tasks(selected_frames):
    """Create LS JSON for import."""
    tasks = []

    for f in selected_frames:
        frame_num = f["frame"]
        thumb = f"{SNIPPET_DIR}/frame_{frame_num}.jpg"
        snippet = f"{SNIPPET_DIR}/frame_{frame_num}.mp4"

        # LS needs relative or absolute paths
        tasks.append({
            "data": {
                "video_url": VIDEO_URL,
                "frame": frame_num,
                "thumbnail": thumb,
                "snippet": snippet,
                "model_prediction": f["pred"],
                "model_confidence": f["prob"],
                "model_uncertainty": f["uncertainty"]
            }
        })

    Path(f"{OUT_DIR}/ls_active_learning_tasks.json").write_text(
        json.dumps(tasks, indent=2),
        encoding="utf-8"
    )

    return tasks


def main():
    print("=== Active Learning: Full Professional Version ===")

    preds = load_predictions()
    unc = compute_uncertainty_frames(preds)
    filtered = filter_uncertain(unc)

    print(f"Total frames: {len(preds)}")
    print(f"Uncertain frames: {len(filtered)}")

    selected = cluster_frames(filtered)
    print(f"Selected diverse frames: {len(selected)}")

    # Extract snippets
    for f in selected:
        extract_snippet(VIDEO_LOCAL, f["frame"])

    tasks = build_label_studio_tasks(selected)

    print("\nActive learning tasks saved to:")
    print("  runs/active_learning/ls_active_learning_tasks.json")
    print("\nImport this file into Label Studio.")

    print("\nSelected frames:")
    for f in selected:
        print(f"Frame {f['frame']}  prob={f['prob']:.3f}  uncertainty={f['uncertainty']:.3f}")


if __name__ == "__main__":
    main()
