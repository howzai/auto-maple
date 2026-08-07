"""Train Auto Maple scene models from the manually reviewed dataset.

The default first-stage workflow is intentionally monster-only.  The main labeler
uses the five-class master schema (monster is class 1), while the first detector
should learn one clean class as quickly as possible.  This script therefore builds
a temporary reviewed-only dataset, remaps ``monster`` from class 1 to class 0,
keeps reviewed empty images as negative examples, trains YOLO, and publishes the
best checkpoint as ``assets/models/classic_scene.pt``.

Later, ``--mode full`` trains directly from the five-class dataset.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "training_dataset"
DEFAULT_MODEL = ROOT / "assets" / "models" / "classic_scene.pt"
MASTER_MONSTER_CLASS = 1


def _image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _label_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def reviewed_train_items(dataset: Path) -> list[tuple[Path, Path]]:
    """Return images the user has actually reviewed in the labeler.

    The labeler writes an empty txt file when a reviewed image contains no target.
    That distinction is important: empty reviewed images are useful negatives,
    while images with no txt file have not been reviewed yet and must not train.
    """
    images_dir = dataset / "images" / "train"
    labels_dir = dataset / "labels" / "train"
    items: list[tuple[Path, Path]] = []
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return items
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = _image_for_stem(images_dir, label_path.stem)
        if image_path is not None:
            items.append((image_path, label_path))
    return items


def _monster_lines(label_path: Path) -> list[str]:
    converted: list[str] = []
    for line in _label_lines(label_path):
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        if class_id != MASTER_MONSTER_CLASS:
            continue
        if not all(0.0 <= value <= 1.0 for value in coords):
            continue
        converted.append("0 " + " ".join(f"{value:.6f}" for value in coords))
    return converted


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    values = list(lines)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def prepare_monster_bootstrap(dataset: Path, seed: int = 20260807) -> tuple[Path, int, int, int]:
    items = reviewed_train_items(dataset)
    if not items:
        raise RuntimeError("No reviewed training images found. Label some images first.")

    rng = random.Random(seed)
    rng.shuffle(items)
    val_count = max(1, int(round(len(items) * 0.15)))
    if len(items) >= 20:
        val_count = max(3, val_count)
    val_count = min(val_count, max(1, len(items) - 1))
    split_items = {
        "val": items[:val_count],
        "train": items[val_count:],
    }

    output = (dataset / "monster_bootstrap").resolve()
    if output.exists():
        shutil.rmtree(output)

    total_boxes = 0
    positive_images = 0
    for split, rows in split_items.items():
        images_out = output / "images" / split
        labels_out = output / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for image_path, label_path in rows:
            shutil.copy2(image_path, images_out / image_path.name)
            lines = _monster_lines(label_path)
            if lines:
                positive_images += 1
                total_boxes += len(lines)
            _write_lines(labels_out / f"{image_path.stem}.txt", lines)

    yaml_path = output / "data.yaml"
    # Ultralytics may resolve a relative `path: .` against the process working
    # directory instead of this YAML file on some Windows versions.  Use the
    # absolute bootstrap directory so images/train and images/val always resolve
    # to the dataset we just created.
    yaml_root = output.as_posix().replace("'", "''")
    yaml_path.write_text(
        f"path: '{yaml_root}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: monster\n",
        encoding="utf-8",
    )
    return yaml_path, len(items), positive_images, total_boxes


def count_full_labels(dataset: Path) -> tuple[int, int]:
    files = 0
    boxes = 0
    labels_root = dataset / "labels"
    for split in ("train", "val", "test"):
        split_dir = labels_root / split
        if not split_dir.is_dir():
            continue
        for label_path in split_dir.glob("*.txt"):
            lines = _label_lines(label_path)
            if lines:
                files += 1
                boxes += len(lines)
    return files, boxes


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Auto Maple classic scene YOLO model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=("monster", "full"), default="monster")
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics auto-size the batch")
    parser.add_argument("--device", default=None, help="Examples: 0, cpu. Default lets Ultralytics choose.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        print(f"[ERROR] Dataset not found: {dataset}")
        print("Run build_dataset.bat first.")
        return 1

    if args.mode == "monster":
        try:
            yaml_path, reviewed, positives, boxes = prepare_monster_bootstrap(dataset, args.seed)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            return 2
        if reviewed < 60 or boxes < 60:
            print(f"[ERROR] Monster bootstrap needs at least 60 reviewed images / 60 monster boxes.")
            print(f"[ERROR] Current: {reviewed} reviewed images / {boxes} monster boxes.")
            return 2
        labeled_files = positives
        data_description = f"monster bootstrap ({reviewed} reviewed, including negatives)"
    else:
        yaml_path = dataset / "data.yaml"
        if not yaml_path.is_file():
            print(f"[ERROR] Dataset config not found: {yaml_path}")
            return 1
        labeled_files, boxes = count_full_labels(dataset)
        if labeled_files < 20 or boxes < 40:
            print(f"[ERROR] Not enough labeled data: {labeled_files} labeled images / {boxes} boxes.")
            return 2
        data_description = "full five-class dataset"

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"[ERROR] Ultralytics is not installed: {exc}")
        print("Run setup_ai.bat first.")
        return 3

    print("\n========================================")
    print("  Auto Maple Classic Scene Training")
    print("========================================")
    print(f"Dataset: {dataset}")
    print(f"Mode: {data_description}")
    print(f"Positive labeled images: {labeled_files}")
    print(f"Monster/YOLO boxes: {boxes}")
    print(f"Training YAML: {yaml_path}")
    print(f"Base model: {args.base_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print()

    model = YOLO(args.base_model)
    train_kwargs = {
        "data": str(yaml_path.resolve()),
        "epochs": max(1, args.epochs),
        "imgsz": max(320, args.imgsz),
        "batch": args.batch,
        "project": str(ROOT / "runs" / "scene"),
        "name": "classic_scene",
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "seed": args.seed,
        "patience": 20,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    save_dir = Path(getattr(results, "save_dir", ROOT / "runs" / "scene" / "classic_scene"))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"[ERROR] Training finished but best.pt was not found at: {best}")
        return 4

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    print("\n[OK] Training completed.")
    print(f"[OK] Best checkpoint: {best}")
    print(f"[OK] Auto Maple model: {output}")
    print("[OK] Restart Auto Maple to load classic_scene.pt.")
    if args.mode == "monster":
        print("[OK] Next step: run auto_label.bat to pre-label the remaining images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
