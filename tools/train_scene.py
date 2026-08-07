"""Train the Auto Maple classic scene detector from training_dataset.

The script validates that YOLO labels exist, starts Ultralytics training, and copies
the best checkpoint to assets/models/classic_scene.pt when training succeeds.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "training_dataset"
DEFAULT_MODEL = ROOT / "assets" / "models" / "classic_scene.pt"


def count_labels(dataset: Path) -> tuple[int, int]:
    files = 0
    boxes = 0
    labels_root = dataset / "labels"
    for split in ("train", "val", "test"):
        split_dir = labels_root / split
        if not split_dir.is_dir():
            continue
        for label_path in split_dir.glob("*.txt"):
            text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            lines = [line for line in text.splitlines() if line.strip()]
            if lines:
                files += 1
                boxes += len(lines)
    return files, boxes


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Auto Maple classic scene YOLO model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics auto-size the batch")
    parser.add_argument("--device", default=None, help="Examples: 0, cpu. Default lets Ultralytics choose.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    yaml_path = dataset / "data.yaml"
    if not yaml_path.is_file():
        print(f"[ERROR] Dataset config not found: {yaml_path}")
        print("Run build_dataset.bat first.")
        return 1

    labeled_files, boxes = count_labels(dataset)
    if labeled_files < 20 or boxes < 40:
        print(f"[ERROR] Not enough labeled data: {labeled_files} labeled images / {boxes} boxes.")
        print("Use label_dataset.bat first. For the first model, aim for at least 200-400 diverse labeled images.")
        return 2

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
    print(f"Labeled images: {labeled_files}")
    print(f"YOLO boxes: {boxes}")
    print(f"Base model: {args.base_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print()

    model = YOLO(args.base_model)
    train_kwargs = {
        "data": str(yaml_path),
        "epochs": max(1, args.epochs),
        "imgsz": max(320, args.imgsz),
        "batch": args.batch,
        "project": str(ROOT / "runs" / "scene"),
        "name": "classic_scene",
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
