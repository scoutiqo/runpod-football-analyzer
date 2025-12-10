#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path
import shutil
import os


def run(cmd, cwd):
    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(cmd))
    print("=" * 80)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"FAILED: {' '.join(cmd)} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: video -> tracks -> events"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", required=True, help="Output folder for analysis")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Temporary internal paths
    tmp_tracks = repo_root / "runs" / "json" / "tmp_tracks.json"
    expected_tracks = repo_root / "runs" / "json" / "tracks_players_ball_simple.json"
    predicted_events = repo_root / "runs" / "json" / "predicted_events_learned.json"

    # 1) Video -> tracks
    run(
        [
            "python",
            "core/run_players_ball_simple.py",
            "--video",
            args.video,
            "--out",
            str(tmp_tracks),
        ],
        cwd=repo_root,
    )

    if not tmp_tracks.exists():
        print(f"ERROR: tracking output not found at {tmp_tracks}")
        sys.exit(1)

    # 2) Make predictor see these tracks (it currently expects tracks_players_ball_simple.json)
    shutil.copy(tmp_tracks, expected_tracks)

    # 3) Tracks -> events (using already-trained model)
    run(["python", "core/predict_events_from_tracks.py"], cwd=repo_root)

    if not predicted_events.exists():
        print(f"ERROR: predicted events not found at {predicted_events}")
        sys.exit(1)

    # 4) Copy final artifacts to output folder
    final_tracks = out_dir / "tracks.json"
    final_events = out_dir / "events.json"

    shutil.copy(tmp_tracks, final_tracks)
    shutil.copy(predicted_events, final_events)

    # 5) Minimal metadata file
    meta = out_dir / "metadata.txt"
    meta.write_text(
        f"video={args.video}\ntracks={final_tracks}\nevents={final_events}\n"
    )

    print("\n" + "=" * 80)
    print(f"Analysis finished.")
    print(f"Tracks: {final_tracks}")
    print(f"Events: {final_events}")
    print(f"Metadata: {meta}")
    print("=" * 80)


if __name__ == "__main__":
    main()
