"""Fast and conservative yellow-player tracking for the classic client.

The local player is represented by a small yellow diamond on the minimap.  This
module keeps the existing template detector as a safety fallback, but adds a
shape-aware yellow detector, same-map continuity, and explicit tracker reset
when the minimap is recalibrated or changes size.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.common import utils


Point = Tuple[float, float]
Candidate = Tuple[Point, float]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _template_candidates(capture, minimap: np.ndarray) -> List[Candidate]:
    """Return only strong local maxima from the upstream player template."""
    template = capture.PLAYER_TEMPLATE if hasattr(capture, "PLAYER_TEMPLATE") else None
    if template is None:
        import src.modules.capture as capture_module

        template = capture_module.PLAYER_TEMPLATE
        threshold = capture_module.PLAYER_MATCH_THRESHOLD
    else:
        threshold = 0.80

    conversion = cv2.COLOR_BGRA2GRAY if minimap.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    gray = cv2.cvtColor(minimap, conversion)
    if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
        return []

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    dilated = cv2.dilate(result, np.ones((3, 3), dtype=np.uint8))
    peak_mask = (result >= threshold) & (result >= dilated - 1e-6)
    ys, xs = np.where(peak_mask)
    if len(xs) == 0:
        return []

    scored = sorted(
        ((float(result[y, x]), int(x), int(y)) for x, y in zip(xs, ys)),
        reverse=True,
    )[:12]

    candidates: List[Candidate] = []
    for score, x, y in scored:
        center = (
            int(round(x + template.shape[1] / 2)),
            int(round(y + template.shape[0] / 2)),
        )
        candidates.append((utils.convert_to_relative(center, minimap), score))
    return candidates


def _yellow_candidates(minimap: np.ndarray) -> List[Candidate]:
    """Find small bright-yellow diamond-like components inside the map canvas."""
    if minimap is None or minimap.size == 0 or minimap.ndim != 3:
        return []

    height, width = minimap.shape[:2]
    if width < 20 or height < 15:
        return []

    # Ignore only a very thin border. Portal icons often live at the far sides,
    # while the local player can legitimately approach an edge, so this margin
    # must remain small.
    margin_x = max(1, round(width * 0.012))
    margin_y = max(1, round(height * 0.012))
    x0, y0 = margin_x, margin_y
    x1, y1 = width - margin_x, height - margin_y
    roi = minimap[y0:y1, x0:x1, :3]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (18, 115, 165), (40, 255, 255))
    yellow_orange = cv2.inRange(hsv, (14, 155, 190), (25, 255, 255))
    mask = cv2.bitwise_or(yellow, yellow_orange)

    # Explicitly remove red markers (other players).
    red_low = cv2.inRange(hsv, (0, 100, 110), (10, 255, 255))
    red_high = cv2.inRange(hsv, (170, 100, 110), (179, 255, 255))
    mask[cv2.bitwise_or(red_low, red_high) > 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Candidate] = []
    map_area = float(width * height)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, component_w, component_h = cv2.boundingRect(contour)
        pixel_area = int(cv2.countNonZero(mask[y:y + component_h, x:x + component_w]))

        # Scale limits with minimap size while keeping sensible absolute bounds.
        min_area = max(2.0, map_area * 0.00008)
        max_area = max(90.0, map_area * 0.012)
        if area < min_area or area > max_area:
            continue
        if not 2 <= component_w <= max(20, round(width * 0.12)):
            continue
        if not 2 <= component_h <= max(20, round(height * 0.18)):
            continue

        aspect = component_w / max(component_h, 1)
        if not 0.45 <= aspect <= 2.2:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, 0.12 * perimeter, True)
        vertices = len(polygon)
        fill = pixel_area / float(max(component_w * component_h, 1))
        compactness = min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))

        component_hsv = hsv[y:y + component_h, x:x + component_w]
        component_mask = mask[y:y + component_h, x:x + component_w] > 0
        if component_hsv.size == 0 or not np.any(component_mask):
            continue
        mean_hue = float(np.mean(component_hsv[:, :, 0][component_mask]))
        saturation = float(np.mean(component_hsv[:, :, 1][component_mask])) / 255.0
        value = float(np.mean(component_hsv[:, :, 2][component_mask])) / 255.0
        if not 14.0 <= mean_hue <= 40.0:
            continue

        moments = cv2.moments(contour)
        if moments["m00"]:
            cx = moments["m10"] / moments["m00"] + x0
            cy = moments["m01"] / moments["m00"] + y0
        else:
            cx = x + component_w / 2 + x0
            cy = y + component_h / 2 + y0

        # A true marker is normally small, bright, saturated and roughly diamond
        # shaped. Shape is a preference rather than a hard requirement because
        # display scaling can make the diamond look irregular.
        vertex_score = 1.0 if 4 <= vertices <= 7 else 0.45
        aspect_score = 1.0 - min(1.0, abs(aspect - 1.0) / 1.2)
        fill_score = 1.0 - min(1.0, abs(fill - 0.58) / 0.58)
        size_target = max(5.0, map_area * 0.0012)
        size_score = 1.0 - min(1.0, abs(pixel_area - size_target) / max(size_target * 2.2, 1.0))
        hue_score = 1.0 - min(1.0, abs(mean_hue - 28.0) / 17.0)

        confidence = min(
            0.995,
            0.18
            + 0.16 * saturation
            + 0.16 * value
            + 0.13 * aspect_score
            + 0.10 * fill_score
            + 0.10 * vertex_score
            + 0.09 * size_score
            + 0.08 * hue_score
            + 0.05 * compactness,
        )
        candidates.append(((cx / width, cy / height), confidence))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:10]


def _deduplicate(candidates: List[Candidate]) -> List[Candidate]:
    candidates.sort(key=lambda item: item[1], reverse=True)
    result: List[Candidate] = []
    for point, score in candidates:
        if all(_distance(point, other[0]) > 0.025 for other in result):
            result.append((point, score))
        if len(result) >= 12:
            break
    return result


def install_classic_player_fallback(capture_class) -> None:
    """Install zero-lag same-map tracking with safe map-change reset."""
    if getattr(capture_class, "_classic_player_patch_installed", False):
        return

    original_accept_position = capture_class._accept_position
    original_reset_tracking = capture_class._reset_tracking

    def reset_classic_tracking(self):
        original_reset_tracking(self)
        self._classic_minimap_shape = None
        self._classic_tracking_position = None
        self._classic_pending_position = None
        self._classic_pending_frames = 0
        self._classic_reacquiring = True

    def bounded_player_candidates(self, minimap: np.ndarray):
        shape = tuple(minimap.shape[:2])
        if getattr(self, "_classic_minimap_shape", None) != shape:
            # A changed ROI size strongly indicates a different map/layout. Do
            # not let proximity to the old coordinate bias the new acquisition.
            self._classic_minimap_shape = shape
            self._classic_tracking_position = None
            self._classic_pending_position = None
            self._classic_pending_frames = 0
            self._classic_reacquiring = True

        yellow = _deduplicate(_yellow_candidates(minimap))
        if yellow:
            self.player_detection_method = "classic-yellow-v2"
            return yellow

        # Preserve the original template path as a conservative fallback.
        template = _deduplicate(_template_candidates(self, minimap))
        self.player_detection_method = "template-fallback" if template else "none"
        return template

    def select_player_candidate(self, candidates):
        if not candidates:
            return None

        method = getattr(self, "player_detection_method", "")
        if method != "classic-yellow-v2":
            if self._filtered_position is None:
                return max(candidates, key=lambda item: item[1])
            return min(candidates, key=lambda item: _distance(item[0], self._filtered_position) - item[1] * 0.02)

        previous: Optional[Point] = getattr(self, "_classic_tracking_position", None)
        reacquiring = bool(getattr(self, "_classic_reacquiring", True))

        if previous is None or reacquiring:
            # On startup/map change, accept a clearly strong candidate instantly
            # for the same responsiveness as before. Ambiguous candidates require
            # two consistent frames before they are allowed to become the anchor.
            strongest = max(candidates, key=lambda item: item[1])
            if strongest[1] >= 0.78 or len(candidates) == 1:
                self._classic_reacquiring = False
                self._classic_pending_position = None
                self._classic_pending_frames = 0
                return strongest

            pending = getattr(self, "_classic_pending_position", None)
            if pending is not None and _distance(strongest[0], pending) <= 0.055:
                self._classic_pending_frames += 1
                self._classic_pending_position = (
                    (pending[0] + strongest[0][0]) / 2.0,
                    (pending[1] + strongest[0][1]) / 2.0,
                )
            else:
                self._classic_pending_position = strongest[0]
                self._classic_pending_frames = 1
            if self._classic_pending_frames >= 2:
                self._classic_reacquiring = False
                return (self._classic_pending_position, strongest[1])
            return None

        # Same map: continuity dominates, while confidence still breaks ties.
        nearby = [item for item in candidates if _distance(item[0], previous) <= 0.22]
        pool = nearby or candidates
        selected = min(pool, key=lambda item: _distance(item[0], previous) - item[1] * 0.055)

        # A very large jump on an unchanged ROI is usually a portal icon or other
        # yellow object. Reacquire instead of publishing an obviously wrong point.
        jump = _distance(selected[0], previous)
        if jump > 0.38 and selected[1] < 0.86:
            self._classic_reacquiring = True
            self._classic_pending_position = selected[0]
            self._classic_pending_frames = 1
            return None
        return selected

    def accept_position_fast(self, position: Point):
        if getattr(self, "player_detection_method", "") == "classic-yellow-v2":
            # Publish the newest verified yellow coordinate on the same capture
            # frame. No EMA or teleport delay is applied after candidate selection.
            self._classic_tracking_position = position
            self._filtered_position = position
            self._pending_jump = None
            self._pending_jump_frames = 0
            return position
        return original_accept_position(self, position)

    capture_class._reset_tracking = reset_classic_tracking
    capture_class._player_candidates = bounded_player_candidates
    capture_class._select_player_candidate = select_player_candidate
    capture_class._accept_position = accept_position_fast
    capture_class._classic_player_patch_installed = True
