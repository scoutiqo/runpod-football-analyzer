import argparse
from pathlib import Path
import cv2


def extract_frames(video_path: Path, out_dir: Path, every_n: int = 50):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            fname = f"{video_path.stem}_f{idx:05d}.jpg"
            cv2.imwrite(str(out_dir / fname), frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"{video_path.name}: saved {saved} / {total} frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="datasets/homography/raw_videos")
    parser.add_argument("--out", default="datasets/homography/frames")
    parser.add_argument("--every-n", type=int, default=50)
    args = parser.parse_args()

    videos_dir = Path(args.videos)
    out_dir = Path(args.out)

    for mp4 in sorted(videos_dir.glob("*.mp4")):
        extract_frames(mp4, out_dir, args.every_n)


if __name__ == "__main__":
    main()
