"""Fast fixed-UI localization for the Traditional Chinese classic client.

The minimap is anchored by fixed UI chrome, never by moving map artwork.  The
locator finds the orange ``地圖`` button (or the pale title-bar fallback), uses
that to determine the panel width, then scans for the first truly full-width
bottom border.  All work is limited to a small upper-left ROI so startup remains
fast and player-marker tracking is not delayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

Bounds = Tuple[Tuple[int, int], Tuple[int, int]]


@dataclass(frozen=True)
class UiRegion:
    name: str
    bounds: Bounds
    confidence: float
    method: str

    @property
    def width(self) -> int:
        return self.bounds[1][0] - self.bounds[0][0]

    @property
    def height(self) -> int:
        return self.bounds[1][1] - self.bounds[0][1]

    def crop(self, frame: np.ndarray) -> np.ndarray:
        (x1, y1), (x2, y2) = self.bounds
        return frame[y1:y2, x1:x2]


@dataclass(frozen=True)
class FixedUiSnapshot:
    minimap_panel: UiRegion
    minimap_canvas: UiRegion
    status_bar: UiRegion
    gameplay_area: UiRegion

    def as_dict(self) -> Dict[str, UiRegion]:
        return {
            "minimap_panel": self.minimap_panel,
            "minimap_canvas": self.minimap_canvas,
            "status_bar": self.status_bar,
            "gameplay_area": self.gameplay_area,
        }


def _clip_bounds(frame: np.ndarray, bounds: Bounds) -> Optional[Bounds]:
    height, width = frame.shape[:2]
    (x1, y1), (x2, y2) = bounds
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1), (x2, y2)


def _find_map_button(roi: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
    """Find the saturated orange/yellow map button in a small top-left ROI."""
    hsv = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2HSV)

    # Two masks cover both the deeper orange and pale yellow client variants.
    orange = cv2.inRange(
        hsv,
        np.array((5, 70, 105), dtype=np.uint8),
        np.array((43, 255, 255), dtype=np.uint8),
    )
    yellow = cv2.inRange(
        hsv,
        np.array((18, 45, 145), dtype=np.uint8),
        np.array((48, 255, 255), dtype=np.uint8),
    )
    mask = cv2.bitwise_or(orange, yellow)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((3, 5), dtype=np.uint8), iterations=1
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    roi_h, roi_w = roi.shape[:2]
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if not (16 <= w <= 125 and 8 <= h <= 48 and area >= 85):
            continue
        if x < 45 or y > min(70, roi_h - 1):
            continue
        fill = area / float(max(w * h, 1))
        aspect = w / float(max(h, 1))
        if fill < 0.24 or not 1.05 <= aspect <= 8.0:
            continue
        # The real button is normally the rightmost compact title-bar component.
        score = 0.48 * fill + 0.22 * min(1.0, aspect / 3.0) + 0.30 * (x / roi_w)
        candidates.append((score, x, y, x + w, y + h))

    if not candidates:
        return None
    score, x1, y1, x2, y2 = max(candidates, key=lambda item: item[0])
    return x1, y1, x2, y2, float(score)


def _fallback_panel_right(gray: np.ndarray) -> Optional[Tuple[int, float]]:
    """Find the pale right panel edge when color-button detection misses."""
    h, w = gray.shape
    top_h = min(h, 78)
    for x in range(min(w - 1, 360), 105, -1):
        strip = gray[:top_h, max(0, x - 2):min(w, x + 2)]
        if strip.size == 0:
            continue
        light = float(np.mean(strip >= 150))
        if light >= 0.42:
            return x, min(1.0, light + 0.18)
    return None


def _find_canvas_top(gray: np.ndarray, panel_right: int, start_y: int, end_y: int) -> Optional[int]:
    x1 = 4
    x2 = max(x1 + 1, panel_right - 5)
    consecutive = 0
    first = None
    for y in range(max(0, start_y), min(end_y, gray.shape[0])):
        row = gray[y:y + 1, x1:x2]
        if row.size == 0:
            continue
        # Map canvas is predominantly dark blue/gray across most of its width.
        dark_fraction = float(np.mean(row < 155))
        if dark_fraction >= 0.68:
            if consecutive == 0:
                first = y
            consecutive += 1
            if consecutive >= 3:
                return first
        else:
            consecutive = 0
            first = None
    return None


def _find_bottom_border(gray: np.ndarray, panel_right: int, canvas_top: int,
                        max_y: int) -> Optional[Tuple[int, float]]:
    """Find a continuous full-width pale border, ignoring map platforms."""
    x1 = 1
    x2 = max(x1 + 1, panel_right - 1)
    run_start = None
    scores = []

    for y in range(canvas_top + 28, min(max_y, gray.shape[0] - 1)):
        row = gray[y:y + 1, x1:x2]
        if row.size == 0:
            continue
        # A real UI border spans nearly the whole panel. Platforms and scenery do
        # not, so require a much higher full-row light fraction than before.
        light = float(np.mean(row >= 145))
        if light >= 0.76:
            if run_start is None:
                run_start = y
                scores = []
            scores.append(light)
            if len(scores) >= 2:
                return run_start, float(np.mean(scores))
        else:
            run_start = None
            scores = []
    return None


def _locate_minimap(frame: np.ndarray) -> Optional[Tuple[UiRegion, UiRegion]]:
    height, width = frame.shape[:2]
    if width < 800 or height < 500:
        return None

    search_w = min(width, max(260, int(width * 0.25)))
    search_h = min(height, max(220, int(height * 0.48)))
    roi = frame[:search_h, :search_w, :3]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    button = _find_map_button(roi)
    if button is not None:
        _, by1, bx2, by2, anchor_score = button
        panel_right = min(search_w, bx2 + 4)
        panel_top = max(0, by1 - 6)
        canvas_search_start = by2 + 2
    else:
        fallback = _fallback_panel_right(gray)
        if fallback is None:
            return None
        panel_right, anchor_score = fallback
        panel_top = 0
        canvas_search_start = 18

    if not 105 <= panel_right <= min(430, int(width * 0.35)):
        return None

    canvas_top = _find_canvas_top(
        gray,
        panel_right,
        start_y=canvas_search_start,
        end_y=min(search_h, canvas_search_start + 165),
    )
    if canvas_top is None:
        return None

    bottom = _find_bottom_border(
        gray,
        panel_right,
        canvas_top=canvas_top,
        max_y=search_h,
    )
    if bottom is None:
        return None
    border_y, border_score = bottom

    panel_bottom = min(height, border_y + 8)
    canvas_bounds = _clip_bounds(
        frame,
        ((4, canvas_top), (panel_right - 5, max(canvas_top + 1, border_y - 2))),
    )
    panel_bounds = _clip_bounds(
        frame,
        ((0, panel_top), (panel_right, panel_bottom)),
    )
    if panel_bounds is None or canvas_bounds is None:
        return None

    (cx1, cy1), (cx2, cy2) = canvas_bounds
    canvas_w, canvas_h = cx2 - cx1, cy2 - cy1
    if canvas_w < 90 or canvas_h < 30:
        return None
    ratio = canvas_w / float(max(canvas_h, 1))
    if not 0.50 <= ratio <= 7.0:
        return None

    # Keep confidence above the backend's safe-accept threshold only when both a
    # fixed title anchor and a true full-width bottom border were found.
    confidence = min(1.0, 0.50 + 0.25 * anchor_score + 0.25 * border_score)
    method = "classic-white-border-anchor-v9"
    return (
        UiRegion("minimap_panel", panel_bounds, confidence, method),
        UiRegion("minimap_canvas", canvas_bounds, confidence, method),
    )


def locate_fixed_ui(frame: np.ndarray) -> Optional[FixedUiSnapshot]:
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None

    located = _locate_minimap(frame)
    if located is None:
        return None
    minimap_panel, minimap_canvas = located

    height, width = frame.shape[:2]
    status_bounds = _clip_bounds(
        frame,
        ((int(width * 0.16), int(height * 0.855)), (int(width * 0.86), height)),
    )
    gameplay_bounds = _clip_bounds(
        frame,
        ((0, int(height * 0.08)), (width, int(height * 0.855))),
    )
    if status_bounds is None or gameplay_bounds is None:
        return None

    return FixedUiSnapshot(
        minimap_panel=minimap_panel,
        minimap_canvas=minimap_canvas,
        status_bar=UiRegion("status_bar", status_bounds, 0.80, "classic-layout"),
        gameplay_area=UiRegion("gameplay_area", gameplay_bounds, 0.85, "classic-layout"),
    )
