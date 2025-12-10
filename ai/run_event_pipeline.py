#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd):
    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(cmd))
    print("=" * 80)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end event pipeline: build dataset -> train model -> predict -> evaluate."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/json/events_learned_v1.json",
        help="Path for canonical events JSON to write (copies predicted_events_learned.json).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    print(f"Repo root: {repo_root}")

    # 1) Build dataset
    run(["python", "training/build_event_dataset.py"], cwd=repo_root)

    # 2) Train model
    run(["python", "training/train_event_model.py"], cwd=repo_root)

    # 3) Predict events for all frames
    run(["python", "core/predict_events_from_tracks.py"], cwd=repo_root)

    # 4) Evaluate on labeled frames (prints metrics)
    run(["python", "core/evaluate_on_labeled_frames.py"], cwd=repo_root)

    # 5) Copy predicted_events_learned.json to the requested canonical output
    src = repo_root / "runs" / "json" / "predicted_events_learned.json"
    dst = repo_root / args.out

    if not src.exists():
        print(f"WARNING: {src} does not exist, cannot copy to {dst}")
        sys.exit(1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    print(f"\nCopied {src} -> {dst}")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
