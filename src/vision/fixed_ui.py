"""Fast fixed-UI localization for the Traditional Chinese classic client.

The minimap is anchored to the upper-left of the captured game client. Detection
uses a small candidate set, but validates the real panel borders so the crop does
not spill into the gameplay scene.
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


def _light_fraction(strip: np.ndarray, threshold: int = 135) -> float:
    if strip.size == 0:
        return 0.0
    return float(np.mean(strip >= threshold))


def _border_scores(gray: np.ndarray, bounds: Bounds) -> Tuple[float, float, float, float]:
    (x1, y1), (x2, y2) = bounds
    top = _light_fraction(gray[y1:min(y1 + 4, y2), x1:x2])
    bottom = _light_fraction(gray[max(y2 - 5, y1):y2, x1:x2])
    left = _light_fraction(gray[y1:y2, x1:min(x1 + 4, x2)])
    right = _light_fraction(gray[y1:y2, max(x2 - 5, x1):x2])
    return top, bottom, left, right


def _canvas_score(frame: np.ndarray, gray: np.ndarray, bounds: Bounds) -> float:
    (x1, y1), (x2, y2) = bounds
    crop = frame[y1:y2, x1:x2, :3]
    gray_crop = gray[y1:y2, x1:x2]
    if crop.size == 0 or gray_crop.size == 0:
        return 0.0

    value = np.max(crop, axis=2)
    dark_fraction = float(np.mean(value < 178))
    edges = cv2.Canny(gray_crop, 40, 120)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    return min(1.0, 0.74 * dark_fraction + 0.26 * min(1.0, edge_density / 0.10))


def _locate_minimap(frame: np.ndarray) -> Optional[Tuple[UiRegion, UiRegion]]:
    height, width = frame.shape[:2]
    if width < 800 or height < 500:
        return None

    gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)

    # In the user's current client the complete panel is about 11-14% of the
    # captured client width. Previous 15-17% candidates included sky/scenery on
    # the right and made the player marker coordinates incorrect.
    x_candidates = (0, 2, 4, 6, 8, 10, 12)
    y_candidates = tuple(sorted(set((
        0, 4, 8, 12, 16, 20, 24,
        int(round(height * 0.025)),
        int(round(height * 0.035)),
    ))))
    width_ratios = (0.105, 0.1125, 0.120, 0.128, 0.136, 0.144)
    height_ratios = (0.180, 0.205, 0.235, 0.280, 0.340, 0.405)

    header_h = max(48, min(86, int(round(height * 0.090))))
    candidates = []

    for x in x_candidates:
        for y in y_candidates:
            for wr in width_ratios:
                panel_w = int(round(width * wr))
                margin_x = max(4, int(round(panel_w * 0.028)))
                for hr in height_ratios:
                    panel_h = int(round(height * hr))
                    panel_bounds = _clip_bounds(frame, ((x, y), (x + panel_w, y + panel_h)))
                    if panel_bounds is None:
                        continue

                    top, bottom, left, right = _border_scores(gray, panel_bounds)
                    # The right and bottom borders are essential. If they are weak,
                    # the candidate usually extends into the moving game scene.
                    if right < 0.30 or bottom < 0.24 or left < 0.22:
                        continue

                    canvas_bounds = _clip_bounds(
                        frame,
                        (
                            (x + margin_x, y + header_h),
                            (x + panel_w - margin_x, y + panel_h - 7),
                        ),
                    )
                    if canvas_bounds is None:
                        continue

                    (cx1, cy1), (cx2, cy2) = canvas_bounds
                    canvas_w, canvas_h = cx2 - cx1, cy2 - cy1
                    if canvas_w < 95 or canvas_h < 32:
                        continue
                    ratio = canvas_w / max(canvas_h, 1)
                    if not 0.62 <= ratio <= 5.8:
                        continue

                    canvas = _canvas_score(frame, gray, canvas_bounds)
                    chrome = 0.15 * top + 0.25 * bottom + 0.20 * left + 0.40 * right
                    # Slightly prefer the smallest valid enclosure. This prevents a
                    # wider candidate from winning merely because the sky is dark.
                    compact_bonus = 0.06 * (1.0 - panel_w / max(width * 0.15, 1))
                    score = 0.58 * chrome + 0.42 * canvas + compact_bonus
                    candidates.append((score, panel_bounds, canvas_bounds))

    if not candidates:
        return None

    score, panel_bounds, canvas_bounds = max(candidates, key=lambda item: item[0])
    if score < 0.38:
        return None

    method = "classic-border-anchor-v7"
    return (
        UiRegion("minimap_panel", panel_bounds, float(score), method),
        UiRegion("minimap_canvas", canvas_bounds, float(score), method),
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
