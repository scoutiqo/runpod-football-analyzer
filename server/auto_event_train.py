#!/usr/bin/env python
import os
import subprocess
import sys

COMMANDS = [
    ["python", "tools/pull_ls_export.py"],
    ["python", "build_event_dataset.py"],
    ["python", "train_event_model.py"],
    ["python", "predict_events_from_tracks.py"],
    ["python", "evaluate_on_labeled_frames.py"],
]

def run(cmd):
    print("\n" + "="*80)
    print("Running:", " ".join(cmd))
    print("="*80)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("FAILED:", " ".join(cmd))
        sys.exit(result.returncode)

if __name__ == "__main__":
    for c in COMMANDS:
        run(c)
