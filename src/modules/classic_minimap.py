"""Fallback minimap calibration for the Traditional Chinese classic client.

The upstream templates target a different regional UI.  This module keeps the
original template calibration as the first choice and only falls back to a
conservative contour-based search in the upper-left part of the game window.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


Bounds = Tuple[Tuple[int, int], Tuple[int, int]]


def _gray(frame: np.ndarray) -> np.ndarray:
    conversion = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(frame, conversion)


def _find_classic_minimap(frame: np.ndarray) -> Optional[Tuple[Bounds, float]]:
    """Locate the classic-client minimap panel in the upper-left screen region."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None

    frame_h, frame_w = frame.shape[:2]
    search_w = min(frame_w, max(320, int(frame_w * 0.48)))
    search_h = min(frame_h, max(260, int(frame_h * 0.48)))
    search = frame[:search_h, :search_w]

    gray = _gray(search)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(joined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 130 or height < 85:
            continue
        if width > search_w * 0.92 or height > search_h * 0.92:
            continue

        aspect = width / max(height, 1)
        if not 1.05 <= aspect <= 2.8:
            continue

        # The classic minimap is anchored very close to the game's upper-left.
        if x > frame_w * 0.18 or y > frame_h * 0.20:
            continue

        region = edges[y:y + height, x:x + width]
        if region.size == 0:
            continue

        edge_density = float(np.count_nonzero(region)) / float(region.size)
        area_ratio = (width * height) / float(frame_w * frame_h)
        proximity = 1.0 - min(1.0, (x + y) / max(frame_w * 0.25, 1.0))

        # Prefer a medium-sized bordered panel close to the upper-left corner.
        size_score = 1.0 - min(1.0, abs(area_ratio - 0.045) / 0.07)
        score = 0.45 * proximity + 0.35 * min(edge_density / 0.18, 1.0) + 0.20 * size_score
        candidates.append((score, x, y, width, height))

    if not candidates:
        return None

    score, x, y, width, height = max(candidates, key=lambda item: item[0])
    if score < 0.42:
        return None

    # The upper portion contains the title and location text.  The actual map
    # canvas occupies the lower portion of the panel in the classic client.
    left = x + max(3, int(width * 0.025))
    top = y + max(28, int(height * 0.38))
    right = x + width - max(3, int(width * 0.025))
    bottom = y + height - max(4, int(height * 0.045))

    if right - left < 100 or bottom - top < 38:
        return None
    if right > frame_w or bottom > frame_h:
        return None

    return ((left, top), (right, bottom)), float(score)


def install_classic_minimap_fallback(capture_class) -> None:
    """Patch Capture calibration while preserving the upstream implementation."""
    if getattr(capture_class, "_classic_minimap_patch_installed", False):
        return

    original_calibrate = capture_class._calibrate_minimap

    def calibrated_with_fallback(self) -> bool:
        if original_calibrate(self):
            self.calibration_method = "template"
            return True

        frame = self.screenshot(delay=0)
        located = _find_classic_minimap(frame) if frame is not None else None
        if located is None:
            original_error = self.last_error or "template calibration failed"
            self.last_error = f"{original_error}; classic minimap fallback found no safe candidate"
            return False

        (top_left, bottom_right), score = located
        x1, y1 = top_left
        x2, y2 = bottom_right
        sample = frame[y1:y2, x1:x2]
        if sample.size == 0:
            self.last_error = "Classic minimap fallback produced an empty crop"
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
            self.calibration_method = "classic-contour"
            self._reset_tracking()

        print(
            "\n[~] Classic minimap calibrated "
            f"(contour score {score:.2f}, {width}x{height}, "
            f"bounds=({x1},{y1})-({x2},{y2}))"
        )
        return True

    capture_class._calibrate_minimap = calibrated_with_fallback
    capture_class._classic_minimap_patch_installed = True
