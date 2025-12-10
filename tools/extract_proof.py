import cv2
import os
import sys
from pathlib import Path

# CONFIG
JOB_ID = "93d6efeb-3432-4294-b643-bdd98472f34f"
VIDEO_PATH = f"tmp_jobs/{JOB_ID}.mp4"
OUTPUT_DIR = "runs/viz/proofs"

# The specific frames you want to verify from your logs
TARGETS = {
    652: "LONG_BALL",
    293: "GOAL",
    39:  "CORNER_TAKEN",
    301: "CORNER_TAKEN",
    150: "CORNER_TAKEN",
    178: "GOAL",
    265: "GOAL",
    506: "SAVE",
    636: "SHORT_PASS",
    152: "CORNER_TAKEN",
    553: "LONG_BALL",
    20:  "CORNER_TAKEN",
    82:  "CORNER_TAKEN",
    25:  "CORNER_TAKEN",
    201: "FOUL_COMMITTED",
    256: "GOAL"
}

def main():
    print(f"🕵️ Extracting Proof for Job: {JOB_ID}")
    
    if not os.path.exists(VIDEO_PATH):
        # Try alternate path
        alt_path = f"runs/videos/{JOB_ID}.mp4"
        if os.path.exists(alt_path):
            print(f"   Found video at: {alt_path}")
            video_path = alt_path
        else:
            print(f"❌ Video file not found: {VIDEO_PATH}")
            print("   Please re-download it or ensure it is in tmp_jobs/")
            return
    else:
        video_path = VIDEO_PATH

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   Video has {total_frames} frames.")

    extracted = 0
    
    for frame_idx, label in TARGETS.items():
        if frame_idx >= total_frames:
            print(f"   ⚠️ Frame {frame_idx} is out of bounds.")
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Draw Label on Image
            text = f"Frame {frame_idx}: {label}"
            cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        2.0, (0, 255, 0), 5)
            
            filename = f"{OUTPUT_DIR}/proof_{frame_idx}_{label}.jpg"
            cv2.imwrite(filename, frame)
            extracted += 1
            print(f"   ✅ Saved: {filename}")
        else:
            print(f"   ❌ Failed to read frame {frame_idx}")

    cap.release()
    print(f"\n🎉 Extracted {extracted} proof images to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
