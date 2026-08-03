"""Fixed UI localization for the Traditional Chinese classic client.

The minimap is located from its fixed panel chrome in the upper-left corner, not
from changing map scenery. This keeps players, monsters and combat effects from
being mistaken for the minimap and supports both wide and square map layouts.
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


def _line_score(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    # The panel border/separators are pale gray, white or light blue.
    return float(np.mean(crop >= 145))


def _canvas_score(frame: np.ndarray, bounds: Bounds) -> float:
    (x1, y1), (x2, y2) = bounds
    crop = frame[y1:y2, x1:x2, :3]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_fraction = float(np.mean(hsv[:, :, 2] < 150))
    edges = cv2.Canny(gray, 35, 120)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    # A real minimap canvas has a dark blue/gray background plus map-detail edges.
    return min(1.0, 0.72 * dark_fraction + 0.28 * min(1.0, edge_density / 0.12))


def _locate_minimap(frame: np.ndarray) -> Optional[Tuple[UiRegion, UiRegion]]:
    """Locate the complete panel first, then derive its inner map canvas.

    Only the upper-left UI chrome is evaluated. Gameplay pixels inside or beside
    the panel are never used as a positional anchor.
    """
    height, width = frame.shape[:2]
    if width < 800 or height < 500:
        return None

    gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    max_x = min(int(width * 0.24), 310)
    max_y = min(int(height * 0.48), 360)
    candidates = []

    # The panel begins very close to the captured client origin. Width and height
    # vary with client scaling and with wide/square map layouts.
    for x in range(0, min(25, max_x - 120), 2):
        for y in range(0, min(55, max_y - 110), 3):
            for panel_w in range(120, max_x - x + 1, 4):
                right = x + panel_w
                if right >= width:
                    continue

                # Panel height can change substantially between maps. Evaluate a
                # broad range but keep the top-left anchor fixed.
                for panel_h in range(105, max_y - y + 1, 6):
                    bottom = y + panel_h
                    if bottom >= height:
                        continue

                    top_border = _line_score(gray, x, y, right, min(y + 4, bottom))
                    left_border = _line_score(gray, x, y, min(x + 4, right), bottom)
                    right_border = _line_score(gray, max(x, right - 4), y, right, bottom)
                    bottom_border = _line_score(gray, x, max(y, bottom - 5), right, bottom)
                    chrome = 0.25 * (top_border + left_border + right_border + bottom_border)
                    if chrome < 0.34:
                        continue

                    # Header/name area is a fixed-height UI band. The actual map
                    # canvas begins below it, independent of the total panel height.
                    header_h = max(55, min(92, int(round(height * 0.105))))
                    margin_x = max(5, int(round(panel_w * 0.035)))
                    canvas_bounds = _clip_bounds(
                        frame,
                        ((x + margin_x, y + header_h), (right - margin_x, bottom - 7)),
                    )
                    if canvas_bounds is None:
                        continue
                    (cx1, cy1), (cx2, cy2) = canvas_bounds
                    canvas_w, canvas_h = cx2 - cx1, cy2 - cy1
                    if canvas_w < 105 or canvas_h < 35:
                        continue
                    if not 0.65 <= canvas_w / max(canvas_h, 1) <= 5.8:
                        continue

                    canvas = _canvas_score(frame, canvas_bounds)
                    # Prefer the smallest panel that cleanly encloses strong chrome
                    # and a plausible dark canvas. This prevents extending into the
                    # gameplay scene when a character or monster passes nearby.
                    compactness = 1.0 - min(1.0, (panel_w * panel_h) / float(max_x * max_y))
                    score = 0.58 * chrome + 0.34 * canvas + 0.08 * compactness
                    candidates.append((score, ((x, y), (right, bottom)), canvas_bounds))

    if not candidates:
        return None

    score, panel_bounds, canvas_bounds = max(candidates, key=lambda item: item[0])
    if score < 0.48:
        return None

    method = "classic-panel-chrome-v4"
    return (
        UiRegion("minimap_panel", panel_bounds, float(score), method),
        UiRegion("minimap_canvas", canvas_bounds, float(score), method),
    )


def locate_fixed_ui(frame: np.ndarray) -> Optional[FixedUiSnapshot]:
    """Return validated fixed UI regions for one captured game frame."""
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

    status_region = UiRegion("status_bar", status_bounds, 0.80, "classic-layout")
    gameplay_region = UiRegion("gameplay_area", gameplay_bounds, 0.85, "classic-layout")

    return FixedUiSnapshot(
        minimap_panel=minimap_panel,
        minimap_canvas=minimap_canvas,
        status_bar=status_region,
        gameplay_area=gameplay_region,
    )
