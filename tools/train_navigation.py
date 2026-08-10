"""Train a separate ladder/platform navigation detector.

Master labeler schema:
    0 player
    1 monster
    2 ladder
    3 platform
    4 obstacle

For the current farming-map bootstrap, player is intentionally excluded because
many existing captures contain a pet that heavily occludes the character.  The
stable monster detector is also kept separate.  This trainer therefore learns
only ladder and platform from images where those classes were manually added.

Compact navigation model schema:
    0 ladder
    1 platform

The best checkpoint is published as assets/models/navigation_scene.pt.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "training_dataset"
DEFAULT_MODEL = ROOT / "assets" / "models" / "navigation_scene.pt"

MASTER_TO_NAV = {
    2: 0,  # ladder
    3: 1,  # platform
}
NAV_NAMES = {
    0: "ladder",
    1: "platform",
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _label_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]


def _navigation_lines(label_path: Path) -> list[str]:
    converted: list[str] = []
    for line in _label_lines(label_path):
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            master_class = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        nav_class = MASTER_TO_NAV.get(master_class)
        if nav_class is None:
            continue
        if not all(0.0 <= value <= 1.0 for value in coords):
            continue
        converted.append(
            f"{nav_class} " + " ".join(f"{value:.6f}" for value in coords)
        )
    return converted


def collect_navigation_items(dataset: Path) -> list[tuple[Path, list[str]]]:
    """Return only images that actually contain ladder/platform annotations."""
    images_dir = dataset / "images" / "train"
    labels_dir = dataset / "labels" / "train"
    items: list[tuple[Path, list[str]]] = []
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return items

    for label_path in sorted(labels_dir.glob("*.txt")):
        lines = _navigation_lines(label_path)
        if not lines:
            continue
        image_path = _image_for_stem(images_dir, label_path.stem)
        if image_path is not None:
            items.append((image_path, lines))
    return items


def prepare_navigation_dataset(
    dataset: Path,
    seed: int,
) -> tuple[Path, int, dict[int, int]]:
    items = collect_navigation_items(dataset)
    if len(items) < 2:
        raise RuntimeError("Not enough ladder/platform-labeled images yet.")

    rng = random.Random(seed)
    rng.shuffle(items)

    val_count = max(1, int(round(len(items) * 0.15)))
    if len(items) >= 20:
        val_count = max(3, val_count)
    val_count = min(val_count, len(items) - 1)

    split_items = {
        "val": items[:val_count],
        "train": items[val_count:],
    }

    output = (dataset / "navigation_bootstrap").resolve()
    if output.exists():
        shutil.rmtree(output)

    class_counts = {0: 0, 1: 0}
    for split, rows in split_items.items():
        images_out = output / "images" / split
        labels_out = output / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for image_path, lines in rows:
            shutil.copy2(image_path, images_out / image_path.name)
            for line in lines:
                try:
                    class_counts[int(line.split()[0])] += 1
                except (ValueError, IndexError, KeyError):
                    pass
            (labels_out / f"{image_path.stem}.txt").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

    yaml_path = output / "data.yaml"
    yaml_root = output.as_posix().replace("'", "''")
    yaml_path.write_text(
        f"path: '{yaml_root}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: ladder\n"
        "  1: platform\n",
        encoding="utf-8",
    )
    return yaml_path, len(items), class_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Auto Maple ladder/platform navigation YOLO model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=None, help="Examples: 0, cpu. Default prefers CUDA when available.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--min-images", type=int, default=30)
    parser.add_argument("--min-boxes", type=int, default=20)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        print(f"[ERROR] Dataset not found: {dataset}")
        print("Run build_dataset.bat first.")
        return 1

    try:
        yaml_path, image_count, counts = prepare_navigation_dataset(dataset, args.seed)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 2

    missing = [NAV_NAMES[c] for c, count in counts.items() if count < args.min_boxes]
    if image_count < args.min_images or missing:
        print("[ERROR] Navigation dataset is still too small for the first training pass.")
        print(f"[ERROR] Navigation-labeled images: {image_count} (need at least {args.min_images})")
        print(
            "[ERROR] Boxes: "
            f"ladder={counts[0]}, platform={counts[1]} "
            f"(need at least {args.min_boxes} each)"
        )
        if missing:
            print("[ERROR] Add more labels for: " + ", ".join(missing))
        return 2

    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        print(f"[ERROR] YOLO/PyTorch is not installed correctly: {exc}")
        print("Run setup_ai.bat first.")
        return 3

    if args.device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("\n========================================")
    print("  Auto Maple Navigation AI Trainer")
    print("========================================")
    print(f"Dataset: {dataset}")
    print(f"Navigation-labeled images: {image_count}")
    print(f"Ladder boxes: {counts[0]}")
    print(f"Platform boxes: {counts[1]}")
    print("Monster-only legacy images: excluded")
    print("Player labels: excluded for this model")
    print(f"Training YAML: {yaml_path}")
    print(f"Base model: {args.base_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training device: {device}\n")

    model = YOLO(args.base_model)
    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=max(1, args.epochs),
        imgsz=max(320, args.imgsz),
        batch=args.batch,
        project=str(ROOT / "runs" / "navigation"),
        name="navigation_scene",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=args.seed,
        patience=25,
        device=device,
    )

    save_dir = Path(
        getattr(results, "save_dir", ROOT / "runs" / "navigation" / "navigation_scene")
    )
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"[ERROR] Training finished but best.pt was not found at: {best}")
        return 4

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)

    print("\n[OK] Navigation training completed.")
    print(f"[OK] Best checkpoint: {best}")
    print(f"[OK] Navigation model: {output}")
    print("[OK] Existing monster model classic_scene.pt was NOT modified.")
    print("[OK] Next: run auto_label_navigation.bat to pre-label ladder/platform boxes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
