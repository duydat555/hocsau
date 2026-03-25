"""
Extracts bounding box crops from the YOLO dataset and organizes them into
an ImageFolder-style structure for training the role classifier.

Output structure:
    classifier_dataset/
        train/
            goalkeeper/  (class 1 in YOLO labels)
            player/      (class 2 in YOLO labels)
        valid/
            goalkeeper/
            player/
"""

from pathlib import Path

import cv2

# Classes to extract: {yolo_class_id: folder_name}
TARGET_CLASSES = {1: "goalkeeper", 2: "player"}

SPLITS = {
    "train": "dataset/football-player.v5i.yolo26/train",
    "valid": "dataset/football-player.v5i.yolo26/valid",
}

OUTPUT_DIR = "classifier_dataset"


def extract_crops(split_name, split_path):
    images_dir = Path(split_path) / "images"
    labels_dir = Path(split_path) / "labels"

    saved = 0
    for label_file in labels_dir.glob("*.txt"):
        img_file = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / (label_file.stem + ext)
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            continue

        frame = cv2.imread(str(img_file))
        if frame is None:
            continue

        h, w = frame.shape[:2]

        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                if cls_id not in TARGET_CLASSES:
                    continue

                cx, cy, bw, bh = map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                # Clamp to frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                    continue

                class_name = TARGET_CLASSES[cls_id]
                out_dir = Path(OUTPUT_DIR) / split_name / class_name
                out_dir.mkdir(parents=True, exist_ok=True)

                out_path = out_dir / f"{label_file.stem}_{saved}.jpg"
                cv2.imwrite(str(out_path), crop)
                saved += 1

    return saved


if __name__ == "__main__":
    total = 0
    for split_name, split_path in SPLITS.items():
        n = extract_crops(split_name, split_path)
        print(f"[{split_name}] Saved {n} crops")
        total += n
    print(f"\nDone. Total crops saved to '{OUTPUT_DIR}': {total}")
