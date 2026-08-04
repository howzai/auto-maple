"""Build a clean training pool from recorded Auto Maple sessions.

This tool is intentionally model-agnostic. It scans ``datasets/session_*`` folders,
filters low-quality sessions, removes near-duplicate frames using perceptual hashes,
keeps event-rich samples, and writes a reproducible manifest under
``training_dataset``. No source session is modified or deleted.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ROOT / "datasets"
DEFAULT_OUTPUT = ROOT / "training_dataset"


@dataclass
class SessionInfo:
    path: Path
    quality: float
    readiness: float
    recommendation: str
    samples: Dict[str, dict]
    event_times: List[float]


@dataclass
class FrameRecord:
    source: Path
    session: str
    source_name: str
    timestamp: float
    sample: dict
    event_rich: bool
    phash: int


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _report_scores(session: Path) -> Tuple[float, float, str]:
    report = _read_json(session / "readiness_report.json", {})
    quality = float(
        report.get("quality_score", report.get("data_quality", report.get("quality", 0))) or 0
    )
    readiness = float(
        report.get("ai_readiness", report.get("readiness_score", report.get("readiness", 0))) or 0
    )
    recommendation = str(report.get("recommendation", ""))

    # Support either 0..1 or 0..100 reports.
    if 0 < quality <= 1:
        quality *= 100
    if 0 < readiness <= 1:
        readiness *= 100
    return quality, readiness, recommendation


def _load_session(path: Path) -> SessionInfo:
    quality, readiness, recommendation = _report_scores(path)
    samples_by_frame: Dict[str, dict] = {}
    for sample in _read_jsonl(path / "samples.jsonl"):
        frame_file = sample.get("frame_file")
        if frame_file:
            samples_by_frame[str(frame_file)] = sample

    event_times: List[float] = []
    for event in _read_jsonl(path / "events.jsonl"):
        try:
            event_times.append(float(event.get("time", 0.0)))
        except (TypeError, ValueError):
            continue

    return SessionInfo(
        path=path,
        quality=quality,
        readiness=readiness,
        recommendation=recommendation,
        samples=samples_by_frame,
        event_times=event_times,
    )


def _dhash(image: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _is_event_rich(timestamp: float, event_times: Sequence[float], window: float = 0.45) -> bool:
    # Session event lists are short enough for this bounded linear check.
    return any(abs(timestamp - value) <= window for value in event_times)


def _session_dirs(root: Path) -> List[Path]:
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith("session_") and (path / "frames").is_dir()
    ) if root.is_dir() else []


def collect_frames(
    sessions: Sequence[SessionInfo],
    min_quality: float,
    duplicate_distance: int,
    event_duplicate_distance: int,
) -> Tuple[List[FrameRecord], List[dict]]:
    accepted: List[FrameRecord] = []
    session_summary: List[dict] = []
    recent_hashes: List[int] = []

    for session in sessions:
        summary = {
            "session": session.path.name,
            "quality": round(session.quality, 2),
            "readiness": round(session.readiness, 2),
            "recommendation": session.recommendation,
            "selected": 0,
            "skipped_low_quality": False,
            "skipped_duplicate": 0,
            "unreadable": 0,
        }

        if session.quality and session.quality < min_quality:
            summary["skipped_low_quality"] = True
            session_summary.append(summary)
            continue

        for frame_path in sorted((session.path / "frames").glob("*.jpg")):
            sample = session.samples.get(frame_path.name, {})
            timestamp = float(sample.get("time", 0.0) or 0.0)
            event_rich = _is_event_rich(timestamp, session.event_times)
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                summary["unreadable"] += 1
                continue

            phash = _dhash(image)
            threshold = event_duplicate_distance if event_rich else duplicate_distance
            is_duplicate = any(_hamming(phash, old_hash) <= threshold for old_hash in recent_hashes[-1200:])
            if is_duplicate:
                summary["skipped_duplicate"] += 1
                continue

            accepted.append(
                FrameRecord(
                    source=frame_path,
                    session=session.path.name,
                    source_name=frame_path.name,
                    timestamp=timestamp,
                    sample=sample,
                    event_rich=event_rich,
                    phash=phash,
                )
            )
            recent_hashes.append(phash)
            summary["selected"] += 1

        session_summary.append(summary)

    return accepted, session_summary


def _split_records(records: List[FrameRecord], seed: int) -> Dict[str, List[FrameRecord]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * 0.80)
    val_end = train_end + int(total * 0.10)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_output(output: Path, splits: Dict[str, List[FrameRecord]], summary: dict, clean: bool) -> None:
    if clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    all_manifest: List[dict] = []
    for split_name, records in splits.items():
        images_dir = output / "images" / split_name
        labels_dir = output / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        split_manifest: List[dict] = []
        for index, record in enumerate(records):
            extension = record.source.suffix.lower() or ".jpg"
            target_name = f"{split_name}_{index:07d}{extension}"
            target = images_dir / target_name
            shutil.copy2(record.source, target)

            row = {
                "split": split_name,
                "image": str(target.relative_to(output)).replace("\\", "/"),
                "source_session": record.session,
                "source_frame": record.source_name,
                "time": round(record.timestamp, 4),
                "event_rich": record.event_rich,
                "keys": record.sample.get("keys", {}),
                "player": record.sample.get("player", {}),
                "scene": record.sample.get("scene"),
            }
            split_manifest.append(row)
            all_manifest.append(row)

        _write_jsonl(output / f"manifest_{split_name}.jsonl", split_manifest)

    _write_jsonl(output / "manifest_all.jsonl", all_manifest)
    (output / "build_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    data_yaml = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: player\n"
        "  1: monster\n"
        "  2: ladder\n"
        "  3: platform\n"
        "  4: obstacle\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")

    readme = (
        "Auto Maple Training Dataset\n"
        "===========================\n\n"
        "這個資料夾由 tools/build_dataset.py 自動產生。\n"
        "原始 session 不會被修改或刪除。\n\n"
        "目前 labels 目錄是空的，下一步要使用標註工具建立 YOLO 標籤。\n"
        "manifest_all.jsonl 保留每張圖片對應的按鍵、角色座標與來源 session。\n"
    )
    (output / "README.txt").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a clean Auto Maple training pool")
    parser.add_argument("--datasets", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-quality", type=float, default=65.0)
    parser.add_argument("--duplicate-distance", type=int, default=5)
    parser.add_argument("--event-duplicate-distance", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    sessions = [_load_session(path) for path in _session_dirs(args.datasets)]
    if not sessions:
        print(f"[!] 找不到可用 Session：{args.datasets}")
        return 1

    frames, session_summary = collect_frames(
        sessions,
        min_quality=args.min_quality,
        duplicate_distance=max(0, args.duplicate_distance),
        event_duplicate_distance=max(0, args.event_duplicate_distance),
    )
    if not frames:
        print("[!] 沒有圖片通過品質與去重條件。")
        return 2

    splits = _split_records(frames, args.seed)
    summary = {
        "format_version": 1,
        "sessions_found": len(sessions),
        "selected_frames": len(frames),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "minimum_quality": args.min_quality,
        "duplicate_distance": args.duplicate_distance,
        "event_duplicate_distance": args.event_duplicate_distance,
        "sessions": session_summary,
    }
    build_output(args.output, splits, summary, clean=not args.no_clean)

    print("\n[~] Dataset Manager 完成")
    print(f"[~] 讀取 Session：{len(sessions)}")
    print(f"[~] 保留圖片：{len(frames)}")
    print(f"[~] Train / Val / Test：{len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")
    print(f"[~] 輸出位置：{args.output}")
    print("[~] 下一步：標註 training_dataset/images 裡的圖片")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
