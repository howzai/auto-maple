"""Pre-label ladder/platform boxes with navigation_scene.pt.

Master dataset class ids remain:
    0 player
    1 monster
    2 ladder
    3 platform
    4 obstacle

The compact navigation model uses:
    0 ladder
    1 platform

Existing monster labels are preserved. Any image that already contains a ladder
or platform label is skipped by default, protecting the user's manually reviewed
navigation annotations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "training_dataset"
DEFAULT_MODEL = ROOT / "assets" / "models" / "navigation_scene.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAV_TO_MASTER = {0: 2, 1: 3}
NAV_MASTER_IDS = {2, 3}


def _images(path: Path):
    return sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ) if path.is_dir() else []


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def _has_navigation_label(lines: list[str]) -> bool:
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id in NAV_MASTER_IDS:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-label ladder/platform boxes")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--max-images", type=int, default=0, help="0 means no limit")
    parser.add_argument("--overwrite-navigation", action="store_true", help="Replace existing ladder/platform labels")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    model_path = args.model.resolve()
    if not model_path.is_file():
        print(f"[ERROR] Navigation model not found: {model_path}")
        print("Run train_navigation_ai.bat first.")
        return 1

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"[ERROR] Ultralytics is not installed: {exc}")
        return 2

    model = YOLO(str(model_path))
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    conf = min(max(float(args.conf), 0.05), 0.99)

    scanned = 0
    written_images = 0
    skipped_reviewed = 0
    ladder_boxes = 0
    platform_boxes = 0

    print("\n========================================")
    print("  Auto Maple Navigation Auto Label")
    print("========================================")
    print(f"Model: {model_path}")
    print(f"Confidence: {conf:.2f}")
    print("Existing monster labels are preserved.")
    if not args.overwrite_navigation:
        print("Images that already contain ladder/platform labels are skipped.")
    print()

    for split in splits:
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split
        labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in _images(images_dir):
            if args.max_images and scanned >= args.max_images:
                break

            label_path = labels_dir / f"{image_path.stem}.txt"
            existing = _read_lines(label_path)
            if _has_navigation_label(existing) and not args.overwrite_navigation:
                skipped_reviewed += 1
                continue

            scanned += 1
            results = model.predict(str(image_path), imgsz=640, conf=conf, verbose=False)
            nav_lines: list[str] = []

            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                height, width = result.orig_shape
                for box in boxes:
                    compact_id = int(box.cls[0])
                    master_id = NAV_TO_MASTER.get(compact_id)
                    if master_id is None:
                        continue
                    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                    cx = ((x1 + x2) / 2.0) / width
                    cy = ((y1 + y2) / 2.0) / height
                    bw = abs(x2 - x1) / width
                    bh = abs(y2 - y1) / height
                    if bw <= 0 or bh <= 0:
                        continue
                    nav_lines.append(f"{master_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    if master_id == 2:
                        ladder_boxes += 1
                    elif master_id == 3:
                        platform_boxes += 1

            if args.overwrite_navigation:
                kept = []
                for line in existing:
                    parts = line.split()
                    try:
                        cid = int(float(parts[0])) if parts else -1
                    except ValueError:
                        cid = -1
                    if cid not in NAV_MASTER_IDS:
                        kept.append(line)
            else:
                kept = existing

            final_lines = kept + nav_lines
            label_path.write_text("\n".join(final_lines) + ("\n" if final_lines else ""), encoding="utf-8")
            written_images += 1

            if written_images % 25 == 0:
                print(
                    f"[~] Pre-labeled {written_images} images | "
                    f"ladder={ladder_boxes} platform={platform_boxes}"
                )

        if args.max_images and scanned >= args.max_images:
            break

    print("\n[OK] Navigation auto-label completed.")
    print(f"[OK] New pre-labeled images: {written_images}")
    print(f"[OK] Ladder boxes: {ladder_boxes}")
    print(f"[OK] Platform boxes: {platform_boxes}")
    print(f"[OK] Preserved manually/navigation-labeled images: {skipped_reviewed}")
    print("[OK] Open label_dataset.bat to review: right-click + Del removes a bad box; drag adds a missing box; Space saves and advances.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
