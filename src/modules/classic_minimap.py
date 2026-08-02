"""Fallback minimap calibration for the Traditional Chinese classic client.

The upstream templates target a different regional UI. This module keeps the
original template calibration as the first choice, then tries contour detection,
and finally uses a conservative upper-left anchored panel search tailored to the
classic Traditional Chinese client.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


Bounds = Tuple[Tuple[int, int], Tuple[int, int]]
Located = Tuple[Bounds, float, str]


def _gray(frame: np.ndarray) -> np.ndarray:
    conversion = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(frame, conversion)


def _valid_bounds(frame: np.ndarray, bounds: Bounds) -> bool:
    (x1, y1), (x2, y2) = bounds
    height, width = frame.shape[:2]
    return (
        0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
        and x2 - x1 >= 100
        and y2 - y1 >= 38
    )


def _find_by_contours(frame: np.ndarray) -> Optional[Located]:
    """Locate a bordered minimap panel using connected contours."""
    frame_h, frame_w = frame.shape[:2]
    search_w = min(frame_w, max(320, int(frame_w * 0.48)))
    search_h = min(frame_h, max(260, int(frame_h * 0.48)))
    search = frame[:search_h, :search_w]

    gray = _gray(search)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 125)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(joined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 140 or height < 100:
            continue
        if width > search_w * 0.92 or height > search_h * 0.92:
            continue

        aspect = width / max(height, 1)
        if not 1.0 <= aspect <= 2.5:
            continue
        if x > frame_w * 0.20 or y > frame_h * 0.24:
            continue

        region = edges[y:y + height, x:x + width]
        if region.size == 0:
            continue

        edge_density = float(np.count_nonzero(region)) / float(region.size)
        proximity = 1.0 - min(1.0, (x + y) / max(frame_w * 0.28, 1.0))
        expected_width = min(210.0, frame_w * 0.20)
        width_score = 1.0 - min(1.0, abs(width - expected_width) / max(expected_width, 1.0))
        score = 0.42 * proximity + 0.38 * min(edge_density / 0.16, 1.0) + 0.20 * width_score
        candidates.append((score, x, y, width, height))

    if not candidates:
        return None

    score, x, y, width, height = max(candidates, key=lambda item: item[0])
    if score < 0.36:
        return None

    # In the classic client the title/location block occupies roughly the top
    # half of the panel, while the actual map canvas is the lower section.
    bounds = (
        (x + max(4, int(width * 0.025)), y + max(45, int(height * 0.48))),
        (x + width - max(4, int(width * 0.025)), y + height - max(5, int(height * 0.045))),
    )
    if not _valid_bounds(frame, bounds):
        return None
    return bounds, float(score), "classic-contour"


def _line_energy(gray: np.ndarray, axis: int) -> np.ndarray:
    """Return normalized gradient energy along rows or columns."""
    if axis == 0:
        gradient = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        energy = np.mean(np.abs(gradient), axis=1)
    else:
        gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        energy = np.mean(np.abs(gradient), axis=0)
    maximum = float(np.max(energy)) if energy.size else 0.0
    return energy / maximum if maximum > 0 else energy


def _find_anchored_panel(frame: np.ndarray) -> Optional[Located]:
    """Find the classic minimap from its stable upper-left screen anchor.

    This method intentionally searches only the first portion of the game window
    and accepts dimensions typical of the Traditional Chinese classic client. It
    avoids selecting chat boxes, hotkey bars, or other large UI panels.
    """
    frame_h, frame_w = frame.shape[:2]
    if frame_w < 640 or frame_h < 400:
        return None

    # Window capture includes the title bar. The minimap starts just below it and
    # remains within the first quarter of the game window.
    search_w = min(frame_w, max(260, int(frame_w * 0.28)))
    search_h = min(frame_h, max(240, int(frame_h * 0.34)))
    search = frame[:search_h, :search_w]
    gray = _gray(search)

    row_energy = _line_energy(gray, axis=0)
    col_energy = _line_energy(gray, axis=1)

    candidates = []
    width_options = range(165, min(255, search_w - 2) + 1, 5)
    height_options = range(120, min(205, search_h - 2) + 1, 5)

    # The panel is nearly flush with the left edge. Search a small range to remain
    # compatible with window borders and DPI scaling.
    for x in range(0, min(45, search_w - 165), 3):
        for y in range(24, min(105, search_h - 120), 3):
            for width in width_options:
                right = x + width
                if right >= search_w:
                    break
                left_border = float(np.mean(col_energy[max(0, x - 1):min(search_w, x + 2)]))
                right_border = float(np.mean(col_energy[max(0, right - 2):min(search_w, right + 1)]))

                for height in height_options:
                    bottom = y + height
                    if bottom >= search_h:
                        break
                    top_border = float(np.mean(row_energy[max(0, y - 1):min(search_h, y + 2)]))
                    bottom_border = float(np.mean(row_energy[max(0, bottom - 2):min(search_h, bottom + 1)]))

                    panel = gray[y:bottom, x:right]
                    if panel.size == 0:
                        continue

                    edge = cv2.Canny(panel, 40, 130)
                    edge_density = float(np.count_nonzero(edge)) / float(edge.size)
                    aspect = width / max(height, 1)
                    aspect_score = 1.0 - min(1.0, abs(aspect - 1.38) / 0.75)
                    anchor_score = 1.0 - min(1.0, (x + max(0, y - 25)) / 120.0)
                    border_score = min(1.0, (left_border + right_border + top_border + bottom_border) / 2.2)
                    density_score = min(1.0, edge_density / 0.17)

                    score = (
                        0.34 * border_score
                        + 0.28 * density_score
                        + 0.23 * anchor_score
                        + 0.15 * aspect_score
                    )
                    if score >= 0.46:
                        candidates.append((score, x, y, width, height))

    if not candidates:
        return None

    score, x, y, width, height = max(candidates, key=lambda item: item[0])

    # The actual map canvas is the lower portion. These ratios match the classic
    # panel while retaining tolerance for different window sizes and DPI scales.
    canvas_left = x + max(5, int(width * 0.035))
    canvas_top = y + max(62, int(height * 0.50))
    canvas_right = x + width - max(5, int(width * 0.035))
    canvas_bottom = y + height - max(7, int(height * 0.055))
    bounds = ((canvas_left, canvas_top), (canvas_right, canvas_bottom))

    if not _valid_bounds(frame, bounds):
        return None
    return bounds, float(score), "classic-anchor"


def find_classic_minimap(frame: np.ndarray) -> Optional[Located]:
    """Try progressively more tolerant classic-client minimap detectors."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None

    located = _find_by_contours(frame)
    if located is not None:
        return located
    return _find_anchored_panel(frame)


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
        located = find_classic_minimap(frame) if frame is not None else None
        if located is None:
            original_error = self.last_error or "template calibration failed"
            self.last_error = f"{original_error}; classic minimap fallback found no safe candidate"
            return False

        (top_left, bottom_right), score, method = located
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
            self.calibration_method = method
            self._reset_tracking()

        print(
            f"\n[~] Classic minimap calibrated using {method} "
            f"(score {score:.2f}, {width}x{height}, "
            f"bounds=({x1},{y1})-({x2},{y2}))"
        )
        return True

    capture_class._calibrate_minimap = calibrated_with_fallback
    capture_class._classic_minimap_patch_installed = True
