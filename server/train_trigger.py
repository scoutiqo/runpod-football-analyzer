# server/train_trigger.py
from pathlib import Path
import json, subprocess

def maybe_queue_training(min_new=10):
    """
    Trigger training when enough new data is available
    """
    stamp = Path("./train_store/.last_count.txt")
    prev = int(stamp.read_text()) if stamp.exists() else 0
    cur = len(list(Path("./datasets/ingest").glob("*/tracks_phoenix.json")))
    if cur - prev >= min_new:
        subprocess.Popen(["bash","-lc","python automl/train_all.py && python automl/evaluate.py && python automl/open_pr.py"])
        stamp.write_text(str(cur))
