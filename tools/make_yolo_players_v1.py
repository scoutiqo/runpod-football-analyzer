import json
import os
from pathlib import Path
import argparse
from collections import defaultdict

def main(ls_json_path: str):
    ls_json_path = Path(ls_json_path)
    out_root = Path("datasets/players_v1")
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"

    if not images_dir.exists():
        raise SystemExit(f"[make_yolo] images dir not found: {images_dir}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[make_yolo] Reading Label Studio export: {ls_json_path}")
    data = json.loads(ls_json_path.read_text())

    if not isinstance(data, list):
        raise SystemExit("[make_yolo] Expected LS export to be a list of tasks")

    all_labels = set()
    tasks_by_image = defaultdict(list)

    for task in data:
        img_url = task.get("data", {}).get("image")
        if not img_url:
            continue

        img_name = os.path.basename(img_url)
        tasks_by_image[img_name].append(task)

        anns = task.get("annotations", [])
        if not anns:
            continue

        for ann in anns:
            for res in ann.get("result", []):
                if res.get("type") != "rectanglelabels":
                    continue
                val = res.get("value", {})
                rect_labels = val.get("rectanglelabels") or []
                if not rect_labels:
                    continue
                all_labels.update(rect_labels)

    if not all_labels:
        raise SystemExit("[make_yolo] No rectanglelabels found in export")

    # Stable class ordering
    class_names = sorted(all_labels)
    print("[make_yolo] Found classes:", class_names)

    class_to_id = {name: i for i, name in enumerate(class_names)}
    print("[make_yolo] Class mapping:", class_to_id)

    # YOLO helpers (LS x,y,width,height are in % of image)
    def ls_to_yolo(val: dict):
        """
        LS:
          x, y, width, height in percentages (top-left, width/height)
        YOLO:
          cx, cy, w, h normalized [0,1]
        """
        x = float(val["x"])
        y = float(val["y"])
        w = float(val["width"])
        h = float(val["height"])

        cx = (x + w / 2.0) / 100.0
        cy = (y + h / 2.0) / 100.0
        w_n = w / 100.0
        h_n = h / 100.0

        return cx, cy, w_n, h_n

    n_images_with_labels = 0
    n_boxes = 0

    for img_name, tasks in tasks_by_image.items():
        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"[make_yolo] WARNING: image not found on disk: {img_path}")
            continue

        label_path = labels_dir / (Path(img_name).stem + ".txt")
        lines = []

        for task in tasks:
            anns = task.get("annotations", [])
            for ann in anns:
                for res in ann.get("result", []):
                    if res.get("type") != "rectanglelabels":
                        continue
                    val = res.get("value", {})
                    rect_labels = val.get("rectanglelabels") or []
                    if not rect_labels:
                        continue

                    lbl = rect_labels[0]
                    if lbl not in class_to_id:
                        continue

                    cls_id = class_to_id[lbl]

                    try:
                        cx, cy, w_n, h_n = ls_to_yolo(val)
                    except Exception as e:
                        print(f"[make_yolo] Bad box in {img_name}: {e}")
                        continue

                    line = f"{cls_id} {cx:.6f} {cy:.6f} {w_n:.6f} {h_n:.6f}"
                    lines.append(line)
                    n_boxes += 1

        if lines:
            label_path.write_text("\n".join(lines) + "\n")
            n_images_with_labels += 1

    print(f"[make_yolo] Wrote labels for {n_images_with_labels} images, {n_boxes} boxes total")

    # Write data.yaml for Ultralytics YOLO
    data_yaml = out_root / "data.yaml"
    yaml_text = "path: ./datasets/players_v1\n"
    yaml_text += "train: images\n"
    yaml_text += "val: images\n"
    yaml_text += "names:\n"
    for i, name in enumerate(class_names):
        yaml_text += f"  {i}: {name}\n"

    data_yaml.write_text(yaml_text)
    print(f"[make_yolo] Wrote {data_yaml}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls-json", required=True, help="Path to Label Studio JSON export")
    args = ap.parse_args()
    main(args.ls_json)
