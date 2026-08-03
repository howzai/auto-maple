"""Fixed UI localization for the Traditional Chinese classic client.

This module only describes and validates stable interface regions. It never sends
input and never decides movement. All coordinates are relative to the captured game
frame so the same layout can tolerate small changes in window borders and DPI.
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


def _texture_score(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    bgr = crop[:, :, :3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 35, 120)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    saturation = float(np.mean(hsv[:, :, 1])) / 255.0
    brightness = float(np.mean(hsv[:, :, 2])) / 255.0
    edge_score = min(1.0, edge_density / 0.18)
    saturation_score = min(1.0, saturation / 0.48)
    brightness_score = 1.0 - min(1.0, abs(brightness - 0.50) / 0.50)
    return 0.56 * edge_score + 0.28 * saturation_score + 0.16 * brightness_score


def _locate_minimap(frame: np.ndarray) -> Optional[Tuple[UiRegion, UiRegion]]:
    """Locate the upper-left minimap panel and its map canvas.

    The search is deliberately bounded to the upper-left portion of the game frame.
    Nearby normalized layouts are scored, then the best safe candidate is retained.
    """
    height, width = frame.shape[:2]
    if width < 800 or height < 500:
        return None

    candidates = []
    for panel_w_ratio in (0.132, 0.142, 0.152, 0.164):
        panel_w = int(round(width * panel_w_ratio))
        for panel_h_ratio in (0.170, 0.185, 0.200, 0.215):
            panel_h = int(round(height * panel_h_ratio))
            for x in (0, 4, 8, 12):
                for y_ratio in (0.028, 0.038, 0.048, 0.058):
                    y = int(round(height * y_ratio))
                    panel_bounds = _clip_bounds(frame, ((x, y), (x + panel_w, y + panel_h)))
                    if panel_bounds is None:
                        continue

                    canvas_bounds = _clip_bounds(
                        frame,
                        (
                            (x + max(5, int(panel_w * 0.035)), y + int(panel_h * 0.48)),
                            (
                                x + panel_w - max(5, int(panel_w * 0.035)),
                                y + panel_h - max(6, int(panel_h * 0.055)),
                            ),
                        ),
                    )
                    if canvas_bounds is None:
                        continue

                    (cx1, cy1), (cx2, cy2) = canvas_bounds
                    if cx2 - cx1 < 110 or cy2 - cy1 < 38:
                        continue
                    score = _texture_score(frame[cy1:cy2, cx1:cx2])
                    candidates.append((score, panel_bounds, canvas_bounds))

    if not candidates:
        return None

    score, panel_bounds, canvas_bounds = max(candidates, key=lambda item: item[0])
    if score < 0.46:
        return None

    return (
        UiRegion("minimap_panel", panel_bounds, float(score), "classic-geometry-v3"),
        UiRegion("minimap_canvas", canvas_bounds, float(score), "classic-geometry-v3"),
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

    # The classic client keeps HP/MP/EXP and hotkeys in the bottom band. This broad
    # crop is intentionally conservative; individual bar readers can refine it.
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
