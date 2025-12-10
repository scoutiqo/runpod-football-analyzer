#!/usr/bin/env python

import cv2
import json
import os
from tqdm import tqdm

from detector import Detector
from tracker_players import PlayerTracker
from ball_tracker import BallTracker
from pipeline import FeatureBuilder


def run_tracking_test(video_path, out_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    print(f"=== TRACKING TEST ON VIDEO: {video_path} ===")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    detector = Detector()
    players_tracker = PlayerTracker()
    ball_tracker = BallTracker()
    feature_builder = FeatureBuilder()

    tracks = []
    frame_id = 0

    pbar = tqdm(total=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect(frame)

        players = players_tracker.update(dets)
        ball = ball_tracker.update(dets)

        features = feature_builder.build(frame_id, players, ball)

        tracks.append({
            "frame": frame_id,
            "players": players,
            "ball": ball,
            "features": features,
        })

        frame_id += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Writing tracking to {out_path}")
    with open(out_path, "w") as f:
        json.dump(tracks, f, indent=2)

    print("=== Tracking test complete ===")
    print(f"Output written to: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out", type=str, default="runs/json/tracking_test.json")

    args = parser.parse_args()
    run_tracking_test(args.video, args.out)
