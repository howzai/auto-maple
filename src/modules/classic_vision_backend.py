"""Classic-client vision backend with optional YOLO and fixed UI localization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.vision.fixed_ui import FixedUiSnapshot, locate_fixed_ui


Bounds = Tuple[Tuple[int, int], Tuple[int, int]]
Located = Tuple[Bounds, float, str]

_MODEL = None
_MODEL_ATTEMPTED = False
RELOCALIZE_INTERVAL_SECONDS = 0.50
RELOCALIZE_PIXEL_TOLERANCE = 3


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yolo():
    global _MODEL, _MODEL_ATTEMPTED
    if _MODEL_ATTEMPTED:
        return _MODEL
    _MODEL_ATTEMPTED = True

    weights = _project_root() / "assets" / "models" / "classic_maple.pt"
    if not weights.is_file():
        return None

    try:
        from ultralytics import YOLO

        _MODEL = YOLO(str(weights))
    except Exception as exc:
        print(f"\n[!] Classic YOLO model could not be loaded: {exc}")
        _MODEL = None
    return _MODEL


def _valid(frame: np.ndarray, bounds: Bounds) -> bool:
    (x1, y1), (x2, y2) = bounds
    height, width = frame.shape[:2]
    return (
        0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
        and 110 <= x2 - x1 <= max(360, int(width * 0.35))
        and 38 <= y2 - y1 <= max(220, int(height * 0.30))
    )


def _find_with_yolo(frame: np.ndarray) -> Optional[Located]:
    model = _load_yolo()
    if model is None:
        return None

    try:
        results = model.predict(frame[:, :, :3], imgsz=640, conf=0.55, verbose=False)
    except Exception:
        return None

    best = None
    for result in results:
        names = result.names
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            label = str(names.get(class_id, class_id)).casefold()
            if label not in {"minimap", "mini_map", "map"}:
                continue
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
            bounds = ((x1, y1), (x2, y2))
            if _valid(frame, bounds) and (best is None or confidence > best[1]):
                best = (bounds, confidence, "yolo-minimap")
    return best


def locate_classic_minimap(frame: np.ndarray) -> Optional[Tuple[Located, Optional[FixedUiSnapshot]]]:
    snapshot = locate_fixed_ui(frame)
    if snapshot is not None:
        region = snapshot.minimap_canvas
        return (region.bounds, region.confidence, region.method), snapshot

    yolo = _find_with_yolo(frame)
    if yolo is not None:
        return yolo, None
    return None


def _bounds_changed(old_bounds: Bounds, new_bounds: Bounds) -> bool:
    old_values = (*old_bounds[0], *old_bounds[1])
    new_values = (*new_bounds[0], *new_bounds[1])
    return any(abs(old - new) > RELOCALIZE_PIXEL_TOLERANCE for old, new in zip(old_values, new_values))


def _apply_location(self, frame, located, ui_snapshot, reset_tracking: bool) -> bool:
    (top_left, bottom_right), score, method = located
    x1, y1 = top_left
    x2, y2 = bottom_right
    sample = frame[y1:y2, x1:x2]
    if sample.size == 0:
        self.last_error = "Classic vision backend produced an empty minimap crop"
        return False

    width = x2 - x1
    height = y2 - y1
    with self._state_lock:
        self._minimap_tl = top_left
        self._minimap_br = bottom_right
        self.minimap_ratio = width / max(height, 1)
        self.minimap_sample = sample.copy()
        self.frame = frame
        self.calibrated = True
        self.last_error = None
        self.calibration_method = method
        self.fixed_ui_snapshot = ui_snapshot
        if reset_tracking:
            self._reset_tracking()

    return True


def install_classic_vision_backend(capture_class) -> None:
    """Replace minimap calibration and keep its bounds synchronized with the UI."""
    if getattr(capture_class, "_classic_vision_backend_installed", False):
        return

    original_capture_and_track = capture_class._capture_and_track

    def calibrate(self) -> bool:
        frame = self.screenshot(delay=0)
        if frame is None:
            return False

        result = locate_classic_minimap(frame)
        if result is None:
            self.last_error = (
                "Classic minimap not found safely. Keep the minimap expanded and "
                "leave the game window at its normal size."
            )
            return False

        located, ui_snapshot = result
        if not _apply_location(self, frame, located, ui_snapshot, reset_tracking=True):
            return False

        (top_left, bottom_right), score, method = located
        x1, y1 = top_left
        x2, y2 = bottom_right
        self._classic_last_relocalize = time.monotonic()
        print(
            f"\n[~] Classic minimap calibrated using {method} "
            f"(score {score:.2f}, {x2 - x1}x{y2 - y1}, "
            f"bounds=({x1},{y1})-({x2},{y2}))"
        )
        if ui_snapshot is not None:
            status = ui_snapshot.status_bar
            gameplay = ui_snapshot.gameplay_area
            print(
                "[~] Fixed UI regions ready: "
                f"status={status.width}x{status.height}, "
                f"gameplay={gameplay.width}x{gameplay.height}"
            )
        return True

    def capture_and_track_with_relocalization(self) -> bool:
        ok = original_capture_and_track(self)
        if not ok or self.frame is None:
            return ok

        now = time.monotonic()
        last = getattr(self, "_classic_last_relocalize", 0.0)
        if now - last < RELOCALIZE_INTERVAL_SECONDS:
            return ok
        self._classic_last_relocalize = now

        result = locate_classic_minimap(self.frame)
        if result is None:
            return ok

        located, ui_snapshot = result
        new_bounds = located[0]
        old_bounds = (self._minimap_tl, self._minimap_br)
        if _bounds_changed(old_bounds, new_bounds):
            if _apply_location(self, self.frame, located, ui_snapshot, reset_tracking=True):
                (x1, y1), (x2, y2) = new_bounds
                print(
                    "\n[~] Minimap layout changed; bounds refreshed "
                    f"to ({x1},{y1})-({x2},{y2})"
                )
        return ok

    capture_class._calibrate_minimap = calibrate
    capture_class._capture_and_track = capture_and_track_with_relocalization
    capture_class._classic_vision_backend_installed = True
