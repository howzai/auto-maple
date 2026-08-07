"""Classic-client vision backend with optional YOLO and stable UI localization."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.vision.fixed_ui import FixedUiSnapshot, locate_fixed_ui
from src.modules.classic_minimap import find_classic_minimap


Bounds = Tuple[Tuple[int, int], Tuple[int, int]]
Located = Tuple[Bounds, float, str]

_MODEL = None
_MODEL_ATTEMPTED = False

# Relocalizing the panel is expensive and does not need to run every few frames.
# The live minimap crop is still refreshed on every captured frame.
RELOCALIZE_INTERVAL_SECONDS = 2.0
RELOCALIZE_CONFIRMATIONS = 3
RELOCALIZE_CLUSTER_TOLERANCE = 8
MAX_SINGLE_STEP_PIXELS = 28
MAX_SIZE_CHANGE_RATIO = 0.22

ANCHORED_RESIZE_TOP_LEFT_TOLERANCE = 18
ANCHORED_RESIZE_MAX_WIDTH_RATIO = 1.35
ANCHORED_RESIZE_MAX_HEIGHT_RATIO = 2.50
MIN_ACCEPTED_SCORE = 0.62


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yolo():
    global _MODEL, _MODEL_ATTEMPTED
    if _MODEL_ATTEMPTED:
        return _MODEL
    _MODEL_ATTEMPTED = True

    candidates = (
        _project_root() / "assets" / "models" / "classic_scene.pt",
        _project_root() / "assets" / "models" / "classic_maple.pt",
    )
    weights = next((path for path in candidates if path.is_file()), None)
    if weights is None:
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
    crop_width = x2 - x1
    crop_height = y2 - y1
    return (
        0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
        and x1 <= max(90, int(width * 0.10))
        and y1 <= max(90, int(height * 0.14))
        and 90 <= crop_width <= max(430, int(width * 0.36))
        and 30 <= crop_height <= max(340, int(height * 0.44))
        and 0.45 <= crop_width / max(crop_height, 1) <= 7.0
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


def locate_classic_minimap(frame: np.ndarray) -> Optional[Tuple[Located, Optional[FixedUiSnapshot]]]:
    """Locate the minimap using multiple independent strategies.

    Fixed UI geometry is the preferred fast path.  If it misses on a particular
    map layout, fall back to the older contour/anchored detector that was already
    proven to work on the classic client.  YOLO remains a final optional fallback.
    """
    snapshot = locate_fixed_ui(frame)
    if snapshot is not None:
        region = snapshot.minimap_canvas
        located = (region.bounds, region.confidence, region.method)
        if region.confidence >= MIN_ACCEPTED_SCORE and _valid(frame, region.bounds):
            return located, snapshot

    legacy = find_classic_minimap(frame)
    if legacy is not None:
        bounds, score, method = legacy
        if _valid(frame, bounds):
            # Legacy detector uses a different confidence scale; its own function
            # already rejects unsafe candidates, so do not impose v10's threshold.
            return (bounds, float(score), f"{method}-fallback"), None

    yolo = _find_with_yolo(frame)
    if yolo is not None and yolo[1] >= MIN_ACCEPTED_SCORE:
        return yolo, None
    return None


def _flat(bounds: Bounds):
    return (*bounds[0], *bounds[1])


def _bounds_distance(a: Bounds, b: Bounds) -> float:
    return max(abs(x - y) for x, y in zip(_flat(a), _flat(b)))


def _size(bounds: Bounds) -> Tuple[int, int]:
    (x1, y1), (x2, y2) = bounds
    return x2 - x1, y2 - y1


def _plausible_transition(old_bounds: Bounds, new_bounds: Bounds) -> bool:
    old_width, old_height = _size(old_bounds)
    new_width, new_height = _size(new_bounds)
    if old_width <= 0 or old_height <= 0:
        return True

    top_left_shift = math.hypot(
        new_bounds[0][0] - old_bounds[0][0],
        new_bounds[0][1] - old_bounds[0][1],
    )
    width_ratio = abs(new_width - old_width) / old_width
    height_ratio = abs(new_height - old_height) / old_height

    if (
        top_left_shift <= MAX_SINGLE_STEP_PIXELS
        and width_ratio <= MAX_SIZE_CHANGE_RATIO
        and height_ratio <= MAX_SIZE_CHANGE_RATIO
    ):
        return True

    if top_left_shift <= ANCHORED_RESIZE_TOP_LEFT_TOLERANCE:
        width_scale = new_width / max(old_width, 1)
        height_scale = new_height / max(old_height, 1)
        return (
            1.0 / ANCHORED_RESIZE_MAX_WIDTH_RATIO
            <= width_scale
            <= ANCHORED_RESIZE_MAX_WIDTH_RATIO
            and 1.0 / ANCHORED_RESIZE_MAX_HEIGHT_RATIO
            <= height_scale
            <= ANCHORED_RESIZE_MAX_HEIGHT_RATIO
        )
    return False


def _publish_frame_heartbeat(self, frame: np.ndarray) -> float:
    now = time.monotonic()
    with self._state_lock:
        self.frame = frame
        self.frame_id += 1
        self.last_frame_time = now
        self.window["width"] = frame.shape[1]
        self.window["height"] = frame.shape[0]
    return now


def _publish_live_minimap(self) -> None:
    """Refresh the GUI preview from the current frame on every capture cycle."""
    frame = self.frame
    if frame is None:
        return
    x1, y1 = self._minimap_tl
    x2, y2 = self._minimap_br
    if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
        return
    live = frame[y1:y2, x1:x2]
    if live.size == 0:
        return
    with self._state_lock:
        self.minimap_sample = live.copy()
        if isinstance(getattr(self, "minimap", None), dict):
            self.minimap["minimap"] = live.copy()


def _apply_location(self, frame, located, ui_snapshot, reset_tracking: bool) -> bool:
    (top_left, bottom_right), score, method = located
    if not _valid(frame, (top_left, bottom_right)):
        return False

    x1, y1 = top_left
    x2, y2 = bottom_right
    sample = frame[y1:y2, x1:x2]
    if sample.size == 0:
        self.last_error = "Classic vision backend produced an empty minimap crop"
        return False

    width = x2 - x1
    height = y2 - y1
    now = time.monotonic()
    with self._state_lock:
        self._minimap_tl = top_left
        self._minimap_br = bottom_right
        self.minimap_ratio = width / max(height, 1)
        self.minimap_sample = sample.copy()
        self.frame = frame
        self.frame_id += 1
        self.last_frame_time = now
        self.window["width"] = frame.shape[1]
        self.window["height"] = frame.shape[0]
        self.calibrated = True
        self.last_error = None
        self.calibration_method = method
        self.fixed_ui_snapshot = ui_snapshot
        if reset_tracking:
            self._reset_tracking()
    return True


def install_classic_vision_backend(capture_class) -> None:
    """Install stable minimap calibration with confirmed, bounded relocalization."""
    if getattr(capture_class, "_classic_vision_backend_installed", False):
        return

    original_capture_and_track = capture_class._capture_and_track

    def calibrate(self) -> bool:
        frame = self.screenshot(delay=0)
        if frame is None:
            return False

        _publish_frame_heartbeat(self, frame)

        result = locate_classic_minimap(frame)
        if result is None:
            self.last_error = (
                "Classic minimap not found safely. Keep the minimap expanded and "
                "leave the game window at its normal size."
            )
            return False

        located, ui_snapshot = result
        if not _apply_location(self, frame, located, ui_snapshot, reset_tracking=True):
            return False

        (top_left, bottom_right), score, method = located
        x1, y1 = top_left
        x2, y2 = bottom_right
        self._classic_last_relocalize = time.monotonic()
        self._classic_pending_bounds = None
        self._classic_pending_count = 0
        print(
            f"\n[~] Classic minimap locked using {method} "
            f"(score {score:.2f}, {x2 - x1}x{y2 - y1}, "
            f"bounds=({x1},{y1})-({x2},{y2}))"
        )
        return True

    def capture_and_track_with_stable_roi(self) -> bool:
        ok = original_capture_and_track(self)
        if not ok or self.frame is None:
            return ok

        _publish_live_minimap(self)

        now = time.monotonic()
        last = getattr(self, "_classic_last_relocalize", 0.0)
        if now - last < RELOCALIZE_INTERVAL_SECONDS:
            return ok
        self._classic_last_relocalize = now

        result = locate_classic_minimap(self.frame)
        if result is None:
            self._classic_pending_bounds = None
            self._classic_pending_count = 0
            return ok

        located, ui_snapshot = result
        new_bounds = located[0]
        old_bounds = (self._minimap_tl, self._minimap_br)

        if _bounds_distance(old_bounds, new_bounds) <= RELOCALIZE_CLUSTER_TOLERANCE:
            self._classic_pending_bounds = None
            self._classic_pending_count = 0
            return ok

        if not _plausible_transition(old_bounds, new_bounds):
            self._classic_pending_bounds = None
            self._classic_pending_count = 0
            return ok

        pending = getattr(self, "_classic_pending_bounds", None)
        if pending is None or _bounds_distance(pending, new_bounds) > RELOCALIZE_CLUSTER_TOLERANCE:
            self._classic_pending_bounds = new_bounds
            self._classic_pending_count = 1
            return ok

        self._classic_pending_count = getattr(self, "_classic_pending_count", 0) + 1
        if self._classic_pending_count < RELOCALIZE_CONFIRMATIONS:
            return ok

        if _apply_location(self, self.frame, located, ui_snapshot, reset_tracking=True):
            (x1, y1), (x2, y2) = new_bounds
            print(
                "\n[~] Confirmed minimap layout change; ROI relocked "
                f"to ({x1},{y1})-({x2},{y2})"
            )

        self._classic_pending_bounds = None
        self._classic_pending_count = 0
        return ok

    capture_class._calibrate_minimap = calibrate
    capture_class._capture_and_track = capture_and_track_with_stable_roi
    capture_class._classic_vision_backend_installed = True
