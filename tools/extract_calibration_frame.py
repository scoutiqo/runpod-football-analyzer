import cv2
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Extract calibration frame from video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frame", type=int, default=200)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(args.frame, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    success, frame = cap.read()
    if not success:
        raise RuntimeError("Could not read calibration frame")

    outfile = out / "calibration_frame.jpg"
    cv2.imwrite(str(outfile), frame)

    print(f"Saved calibration frame: {outfile} (frame {idx}/{total})")


if __name__ == "__main__":
    main()
