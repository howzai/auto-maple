"""Pre-label unreviewed Auto Maple images with the trained monster detector.

The master labeler uses class id 1 for ``monster``.  The first bootstrap model is a
single-class YOLO model whose class id 0 is named ``monster``.  This tool converts
predictions back into the master five-class schema and, by default, NEVER
OVERWRITES any txt file already created by the human labeler.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "training_dataset"
DEFAULT_MODEL = ROOT / "assets" / "models" / "classic_scene.pt"
MASTER_MONSTER_CLASS = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images(path: Path):
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ) if path.is_dir() else []


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-label unreviewed Auto Maple monster images")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--conf", type=float, default=0.72, help="Minimum confidence for pre-label boxes")
    parser.add_argument("--max-images", type=int, default=0, help="0 means no limit")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing labels (not recommended)")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    model_path = args.model.resolve()
    if not model_path.is_file():
        print(f"[ERROR] Model not found: {model_path}")
        print("Train the first model with train_ai.bat first.")
        return 1

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"[ERROR] Ultralytics is not installed: {exc}")
        print("Run setup_ai.bat first.")
        return 2

    model = YOLO(str(model_path))
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    conf = min(max(float(args.conf), 0.05), 0.99)

    scanned = 0
    written = 0
    skipped_reviewed = 0
    total_boxes = 0

    print("\n========================================")
    print("  Auto Maple Monster Auto Label")
    print("========================================")
    print(f"Model: {model_path}")
    print(f"Confidence: {conf:.2f}")
    print("Existing human-reviewed labels will be preserved." if not args.overwrite else "WARNING: overwrite mode enabled.")
    print()

    for split in splits:
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)
        for image_path in _images(images_dir):
            if args.max_images and scanned >= args.max_images:
                break
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists() and not args.overwrite:
                skipped_reviewed += 1
                continue

            scanned += 1
            results = model.predict(str(image_path), imgsz=640, conf=conf, verbose=False)
            lines: list[str] = []
            for result in results:
                boxes = getattr(result, "boxes", None)
                names = getattr(result, "names", {})
                if boxes is None:
                    continue
                height, width = result.orig_shape
                for box in boxes:
                    class_id = int(box.cls[0])
                    label = str(names.get(class_id, class_id)).casefold().strip()
                    if label not in {"monster", "mob", "enemy", "0"} and len(names) > 1:
                        continue
                    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                    cx = ((x1 + x2) / 2.0) / width
                    cy = ((y1 + y2) / 2.0) / height
                    bw = abs(x2 - x1) / width
                    bh = abs(y2 - y1) / height
                    if bw <= 0 or bh <= 0:
                        continue
                    lines.append(f"{MASTER_MONSTER_CLASS} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Write even an empty file.  In the labeler this means the image has
            # been pre-reviewed as a likely negative; the user can still add boxes.
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written += 1
            total_boxes += len(lines)
            if written % 25 == 0:
                print(f"[~] Auto-labeled {written} images / {total_boxes} monster boxes")

        if args.max_images and scanned >= args.max_images:
            break

    print("\n[OK] Auto-label pass completed.")
    print(f"[OK] New pre-labeled images: {written}")
    print(f"[OK] New monster boxes: {total_boxes}")
    print(f"[OK] Preserved existing reviewed images: {skipped_reviewed}")
    print("[OK] Open label_dataset.bat and review/correct the generated boxes before the next training round.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
