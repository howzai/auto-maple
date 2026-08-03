"""Classic-client vision backend with optional YOLO and safe geometric fallback.

The Traditional Chinese classic client uses a stable upper-left minimap panel that
is different from the upstream regional templates. This module prefers a trained
Ultralytics model when ``assets/models/classic_maple.pt`` exists. Until weights are
available, it uses a bounded geometric search and rejects weak candidates rather
than allowing automation to act on an unrelated part of the frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


Bounds = Tuple[Tuple[int, int], Tuple[int, int]]
Located = Tuple[Bounds, float, str]

_MODEL = None
_MODEL_ATTEMPTED = False


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


def _candidate_score(frame: np.ndarray, bounds: Bounds) -> float:
    (x1, y1), (x2, y2) = bounds
    crop = frame[y1:y2, x1:x2, :3]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 35, 120)

    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    saturation = float(np.mean(hsv[:, :, 1])) / 255.0
    brightness = float(np.mean(hsv[:, :, 2])) / 255.0

    # Classic minimaps contain a detailed scene with many short edges and moderate
    # saturation. Blank title bars, sky, chat boxes and black margins score poorly.
    edge_score = min(1.0, edge_density / 0.18)
    saturation_score = min(1.0, saturation / 0.48)
    brightness_score = 1.0 - min(1.0, abs(brightness - 0.50) / 0.50)

    return 0.56 * edge_score + 0.28 * saturation_score + 0.16 * brightness_score


def _find_geometric(frame: np.ndarray) -> Optional[Located]:
    """Locate the classic minimap canvas from its stable upper-left layout."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None

    height, width = frame.shape[:2]
    if width < 800 or height < 500:
        return None

    # Ratios are based on the classic client panel. Multiple nearby sizes are
    # evaluated so DPI scaling and small window-size changes remain supported.
    candidates = []
    for panel_w_ratio in (0.132, 0.142, 0.152, 0.164):
        panel_w = int(round(width * panel_w_ratio))
        for panel_h_ratio in (0.170, 0.185, 0.200, 0.215):
            panel_h = int(round(height * panel_h_ratio))
            for x in (0, 4, 8, 12):
                for y_ratio in (0.028, 0.038, 0.048, 0.058):
                    y = int(round(height * y_ratio))
                    # The map canvas occupies the lower half of the minimap panel.
                    left = x + max(5, int(panel_w * 0.035))
                    top = y + int(panel_h * 0.48)
                    right = x + panel_w - max(5, int(panel_w * 0.035))
                    bottom = y + panel_h - max(6, int(panel_h * 0.055))
                    bounds = ((left, top), (right, bottom))
                    if not _valid(frame, bounds):
                        continue
                    score = _candidate_score(frame, bounds)
                    candidates.append((score, bounds))

    if not candidates:
        return None

    score, bounds = max(candidates, key=lambda item: item[0])
    if score < 0.46:
        return None
    return bounds, float(score), "classic-geometry-v2"


def locate_classic_minimap(frame: np.ndarray) -> Optional[Located]:
    return _find_with_yolo(frame) or _find_geometric(frame)


def install_classic_vision_backend(capture_class) -> None:
    """Replace minimap calibration with the classic-client vision backend."""
    if getattr(capture_class, "_classic_vision_backend_installed", False):
        return

    def calibrate(self) -> bool:
        frame = self.screenshot(delay=0)
        if frame is None:
            return False

        located = locate_classic_minimap(frame)
        if located is None:
            self.last_error = (
                "Classic minimap not found safely. Keep the minimap expanded and "
                "leave the game window at a normal size."
            )
            return False

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
            self._reset_tracking()

        print(
            f"\n[~] Classic vision minimap calibrated using {method} "
            f"(score {score:.2f}, {width}x{height}, "
            f"bounds=({x1},{y1})-({x2},{y2}))"
        )
        return True

    capture_class._calibrate_minimap = calibrate
    capture_class._classic_vision_backend_installed = True
