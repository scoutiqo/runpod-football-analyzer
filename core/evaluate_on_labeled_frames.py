#!/usr/bin/env python
"""
Evaluate per-frame predictions on the labeled frames from Label Studio.

Reads:
  - runs/json/ls_export_project1.json
  - runs/json/predicted_events.json
"""

import json
import os
from typing import Any, Dict, List, Tuple

from sklearn.metrics import classification_report, confusion_matrix

LS_EXPORT_PATH = "runs/json/ls_export_project1.json"
PREDS_PATH = "runs/json/predicted_events.json"


def _load_ls_export(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "tasks" in data and isinstance(data["tasks"], list):
            return data["tasks"]
        if "results" in data and isinstance(data["results"], list):
            return data["results"]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected LS export format in {path!r}")


def _extract_events_from_task(task: Dict[str, Any]) -> List[Tuple[int, str]]:
    """
    Extract (frame, label) for TimelineLabels format:
      {
        "ranges": [{"start": 52, "end": 52}],
        "timelinelabels": ["pass"]
      }
    """
    out = []
    annotations = task.get("annotations") or []
    for ann in annotations:
        results = ann.get("result") or []
        for res in results:
            val = res.get("value", {})

            # Only care about TimelineLabels format
            if "ranges" not in val or "timelinelabels" not in val:
                continue

            ranges = val["ranges"]
            labels = val["timelinelabels"]

            if not ranges or not labels:
                continue

            start_frame = ranges[0].get("start")
            raw_label = labels[0]

            if start_frame is None or raw_label is None:
                continue

            out.append((int(start_frame), str(raw_label).strip().lower()))

    return out


def main():
    if not os.path.exists(LS_EXPORT_PATH):
        raise SystemExit(f"{LS_EXPORT_PATH} not found.")
    if not os.path.exists(PREDS_PATH):
        raise SystemExit(f"{PREDS_PATH} not found. Run predict_events_from_tracks.py first.")

    tasks = _load_ls_export(LS_EXPORT_PATH)
    events: List[Tuple[int, str]] = []
    for t in tasks:
        events.extend(_extract_events_from_task(t))
    if not events:
        print("No labeled events found in LS export.")
        return

    LABEL_CANONICAL_MAP = {
        "pass": "pass",
        "successful_pass": "pass",
        "long_pass": "pass",
        "failed_pass": "pass",
        "ball_loss": "ball_loss",
        "turnover": "ball_loss",
        "duel": "duel",
        "tackle": "duel",
    }

    canonical_events: List[Tuple[int, str]] = []
    for frame_idx, raw in events:
        canon = LABEL_CANONICAL_MAP.get(raw)
        if canon is None:
            continue
        canonical_events.append((frame_idx, canon))

    if not canonical_events:
        print("No labeled events left after canonical mapping.")
        return

    # Load predictions
    with open(PREDS_PATH, "r") as f:
        preds = json.load(f)
    pred_by_frame = {int(d["frame"]): d for d in preds}

    y_true: List[str] = []
    y_pred: List[str] = []

    print("Frame  TrueLabel         PredLabel         Prob")
    print("-----  ----------------  ----------------  ------")
    for frame_idx, canon_label in canonical_events:
        pred = pred_by_frame.get(frame_idx)
        if pred is None:
            continue
        y_true.append(canon_label)
        y_pred.append(str(pred["label"]))
        print(
            f"{frame_idx:5d}  {canon_label:<16}  {str(pred['label']):<16}  {float(pred.get('prob', 0.0)):.3f}"
        )

    if not y_true:
        print("No overlap between labeled frames and predictions.")
        return

    labels = sorted(set(y_true + y_pred))
    print("\nClassification report on labeled frames (canonical labels):")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print(labels)
    print(cm)


if __name__ == "__main__":
    main()
