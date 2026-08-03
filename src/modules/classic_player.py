"""Bounded player-marker detection for the Traditional Chinese classic client."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from src.common import utils
from src.detection import detection


Point = Tuple[float, float]
Candidate = Tuple[Point, float]


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


def _deduplicate(candidates: List[Candidate]) -> List[Candidate]:
    candidates.sort(key=lambda item: item[1], reverse=True)
    result: List[Candidate] = []
    for point, score in candidates:
        if all(
            ((point[0] - other[0][0]) ** 2 + (point[1] - other[0][1]) ** 2) ** 0.5 > 0.025
            for other in result
        ):
            result.append((point, score))
        if len(result) >= 12:
            break
    return result


def install_classic_player_fallback(capture_class) -> None:
    """Use the yellow self-marker as the authoritative, low-latency position."""
    if getattr(capture_class, "_classic_player_patch_installed", False):
        return

    original_accept_position = capture_class._accept_position

    def bounded_player_candidates(self, minimap: np.ndarray):
        classic = detection.classic_player_candidates(minimap)

        # Yellow is the local player. Red dots are other players and are already
        # rejected by classic_player_candidates(). Never mix yellow candidates
        # with the legacy grayscale template, which can lock onto static scenery.
        if classic:
            self.player_detection_method = "classic-yellow"
            return _deduplicate(classic)

        template = _template_candidates(self, minimap)
        self.player_detection_method = "template-fallback" if template else "none"
        return _deduplicate(template)

    def accept_position_fast(self, position: Point):
        if getattr(self, "player_detection_method", "") == "classic-yellow":
            # The yellow diamond is small, unique and already shape/color filtered.
            # Track it immediately instead of applying the old slow EMA and
            # multi-frame teleport confirmation intended for noisy templates.
            previous = getattr(self, "_filtered_position", None)
            if previous is None:
                filtered = position
            else:
                alpha = 0.88
                filtered = (
                    alpha * position[0] + (1.0 - alpha) * previous[0],
                    alpha * position[1] + (1.0 - alpha) * previous[1],
                )
            self._filtered_position = filtered
            self._pending_jump = None
            self._pending_jump_frames = 0
            return filtered
        return original_accept_position(self, position)

    capture_class._player_candidates = bounded_player_candidates
    capture_class._accept_position = accept_position_fast
    capture_class._classic_player_patch_installed = True
