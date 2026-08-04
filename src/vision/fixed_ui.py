"""Fast fixed-UI localization for the Traditional Chinese classic client.

The classic minimap is not located from the moving map artwork.  Instead, this
module finds the fixed orange/yellow ``地圖`` button in the title bar, uses it as
the panel's right-hand anchor, and then follows the panel chrome down to the real
bottom border.  The resulting crop therefore remains stable when monsters,
players, scenery, or combat effects move behind the panel.
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


def _find_map_button(frame: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
    """Find the orange/yellow map button in the upper-left title bar."""
    height, width = frame.shape[:2]
    search_w = min(width, max(260, int(width * 0.24)))
    search_h = min(height, max(52, int(height * 0.10)))
    roi = frame[:search_h, :search_w, :3]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # The classic client's map button is a saturated orange/yellow rectangle.
    mask = cv2.inRange(hsv, np.array((8, 105, 125), dtype=np.uint8),
                      np.array((38, 255, 255), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((3, 5), dtype=np.uint8), iterations=1)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if not (18 <= w <= 105 and 9 <= h <= 42 and area >= 120):
            continue
        if y > 38 or x < 55:
            continue
        fill = area / float(max(w * h, 1))
        aspect = w / float(max(h, 1))
        if not (1.25 <= aspect <= 6.5 and fill >= 0.34):
            continue
        # Prefer the rightmost, well-filled title-bar rectangle.
        score = 0.55 * fill + 0.25 * min(1.0, aspect / 3.0) + 0.20 * (x / search_w)
        candidates.append((score, x, y, x + w, y + h))

    if not candidates:
        return None
    score, x1, y1, x2, y2 = max(candidates, key=lambda item: item[0])
    return x1, y1, x2, y2, float(score)


def _first_dark_run(frame: np.ndarray, x1: int, x2: int, start_y: int,
                    end_y: int) -> Optional[int]:
    """Find where the actual dark minimap canvas begins below the title area."""
    value = np.max(frame[:, :, :3], axis=2)
    inner_x1 = min(x2 - 1, x1 + 5)
    inner_x2 = max(inner_x1 + 1, x2 - 5)
    consecutive = 0
    first = None
    for y in range(max(0, start_y), min(end_y, frame.shape[0])):
        row = value[y:y + 1, inner_x1:inner_x2]
        dark_fraction = float(np.mean(row < 175)) if row.size else 0.0
        if dark_fraction >= 0.58:
            if consecutive == 0:
                first = y
            consecutive += 1
            if consecutive >= 3:
                return first
        else:
            consecutive = 0
            first = None
    return None


def _find_bottom_border(frame: np.ndarray, x1: int, x2: int, canvas_top: int,
                        max_y: int) -> Optional[Tuple[int, float]]:
    """Find the first pale horizontal border below the dark map canvas."""
    gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    inner_x1 = min(x2 - 1, x1 + 2)
    inner_x2 = max(inner_x1 + 1, x2 - 2)
    run_start = None
    run_scores = []

    for y in range(canvas_top + 24, min(max_y, frame.shape[0] - 1)):
        row = gray[y:y + 1, inner_x1:inner_x2]
        light = float(np.mean(row >= 145)) if row.size else 0.0
        if light >= 0.48:
            if run_start is None:
                run_start = y
                run_scores = []
            run_scores.append(light)
            if len(run_scores) >= 2:
                return run_start, float(np.mean(run_scores))
        else:
            run_start = None
            run_scores = []
    return None


def _locate_minimap(frame: np.ndarray) -> Optional[Tuple[UiRegion, UiRegion]]:
    height, width = frame.shape[:2]
    if width < 800 or height < 500:
        return None

    button = _find_map_button(frame)
    if button is None:
        return None
    bx1, by1, bx2, by2, button_score = button

    # The title begins a few pixels above the button and the button's right edge is
    # immediately inside the panel's right border.  The panel itself is anchored
    # to the client upper-left, so no moving gameplay pixels are used as anchors.
    panel_left = 0
    panel_top = max(0, by1 - 5)
    panel_right = min(width, bx2 + 4)
    if panel_right < 105 or panel_right > min(420, int(width * 0.34)):
        return None

    search_bottom = min(height, max(190, int(height * 0.46)))
    canvas_top = _first_dark_run(
        frame,
        panel_left,
        panel_right,
        start_y=by2 + 3,
        end_y=min(search_bottom, by2 + 145),
    )
    if canvas_top is None:
        return None

    bottom = _find_bottom_border(
        frame,
        panel_left,
        panel_right,
        canvas_top=canvas_top,
        max_y=search_bottom,
    )
    if bottom is None:
        return None
    border_y, border_score = bottom

    panel_bottom = min(height, border_y + 7)
    canvas_left = panel_left + 4
    canvas_right = panel_right - 5
    canvas_bottom = max(canvas_top + 1, border_y - 2)

    panel_bounds = _clip_bounds(
        frame, ((panel_left, panel_top), (panel_right, panel_bottom))
    )
    canvas_bounds = _clip_bounds(
        frame, ((canvas_left, canvas_top), (canvas_right, canvas_bottom))
    )
    if panel_bounds is None or canvas_bounds is None:
        return None

    (cx1, cy1), (cx2, cy2) = canvas_bounds
    canvas_w, canvas_h = cx2 - cx1, cy2 - cy1
    if canvas_w < 90 or canvas_h < 30:
        return None
    ratio = canvas_w / float(max(canvas_h, 1))
    if not 0.55 <= ratio <= 6.5:
        return None

    confidence = min(1.0, 0.58 * button_score + 0.42 * border_score)
    method = "classic-title-border-anchor-v8"
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
