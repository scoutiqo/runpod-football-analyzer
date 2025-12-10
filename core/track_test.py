from ultralytics import YOLO

model = YOLO(os.getenv("MODEL_PATH","ai/models/detector_v1.pt"))
model.track(
    source="/workspace/videos/full_match_test.mp4",
    tracker="bytetrack.yaml",
    imgsz=1280,
    conf=0.25,
    iou=0.6,
    persist=True,
    save=True,
    device=0,
    name="track_v1_test"
)
