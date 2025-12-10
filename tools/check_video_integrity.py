import cv2
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python tools/check_video_integrity.py <path_to_video>")
    sys.exit(1)

fname = sys.argv[1]
if not os.path.exists(fname):
    print(f"❌ File not found: {fname}")
    sys.exit(1)

# 1. Check Size
size_mb = os.path.getsize(fname) / (1024*1024)
print(f"📦 File Size: {size_mb:.2f} MB")

# 2. Check Metadata
cap = cv2.VideoCapture(fname)
fps = cap.get(cv2.CAP_PROP_FPS)
meta_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"ℹ️  Metadata says: {meta_frames} frames ({meta_frames/fps:.1f} sec)")

# 3. Count REAL Frames (The Truth)
print("🔎 Scanning frames...")
count = 0
while True:
    ret, _ = cap.read()
    if not ret: break
    count += 1
    if count % 1000 == 0: print(f"   Read {count} frames...")

print(f"✅ Actual Readable Frames: {count}")
if count < meta_frames:
    print(f"🚨 CORRUPTION DETECTED: Missing {meta_frames - count} frames.")
else:
    print("✅ Video file is healthy.")
