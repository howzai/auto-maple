"""Classic-client vision backend with optional YOLO and stable UI localization.

This module deliberately treats the minimap ROI as long-lived state.  Once a safe
ROI has been found we keep refreshing that crop every WGC frame and only replace
it after a new location has been observed consistently.  A short WGC stall must
not destroy a previously good ROI; the last good bounds are reused while the
locator recovers in the background.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.vision.fixed_ui import FixedUiSnapshot, locate_fixed_ui


Bounds = Tuple[Tuple[int, int], Tuple[int, int]]
Located = Tuple[Bounds, float, str]

_MODEL = None
_MODEL_ATTEMPTED = False

# Expensive full-panel localization is intentionally slow while the current ROI
# is healthy.  This prevents animated map content from making the crop jump.
RELOCALIZE_INTERVAL_SECONDS = 3.0
RELOCALIZE_CONFIRMATIONS = 4
RELOCALIZE_FAST_CONFIRMATIONS = 2
RELOCALIZE_CLUSTER_TOLERANCE = 7
RELOCALIZE_MIN_PENDING_SECONDS = 1.0
RELOCALIZE_COOLDOWN_SECONDS = 5.0
MAX_SINGLE_STEP_PIXELS = 28
MAX_SIZE_CHANGE_RATIO = 0.22

ANCHORED_RESIZE_TOP_LEFT_TOLERANCE = 18
ANCHORED_RESIZE_MAX_WIDTH_RATIO = 1.35
ANCHORED_RESIZE_MAX_HEIGHT_RATIO = 2.50
MIN_ACCEPTED_SCORE = 0.62

# A previously validated ROI may be reused after a WGC/watchdog reset.  During
# recovery the full locator is still retried in the background; this merely keeps
# capture FPS and the GUI alive instead of falling into a calibration dead-loop.
RECOVERY_RETRY_INTERVAL_SECONDS = 0.35
RECOVERY_LOCALIZER_MISS_LIMIT = 18
MIN_ROI_EDGE_DENSITY = 0.004
MAX_ROI_EDGE_DENSITY = 0.55


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yolo():
    global _MODEL, _MODEL_ATTEMPTED
    if _MODEL_ATTEMPTED:
        return _MODEL
    _MODEL_ATTEMPTED = True

    # New training pipeline writes classic_scene.pt.  Keep the old name as a
    # compatibility fallback for earlier local checkouts.
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
        and 110 <= crop_width <= max(380, int(width * 0.36))
        and 38 <= crop_height <= max(320, int(height * 0.42))
        and 0.72 <= crop_width / max(crop_height, 1) <= 5.5
    )


def _crop_looks_alive(frame: np.ndarray, bounds: Bounds) -> bool:
    """Cheap guard used only for reusing an already validated ROI.

    We do not try to re-detect the panel here.  The goal is simply to reject an
    empty/flat crop after a resize while allowing ordinary animated minimap
    content to change freely.
    """
    if not _valid(frame, bounds):
        return False
    (x1, y1), (x2, y2) = bounds
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    conversion = cv2.COLOR_BGRA2GRAY if crop.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    gray = cv2.cvtColor(crop, conversion)
    if gray.size == 0 or float(np.std(gray)) < 5.0:
        return False
    edges = cv2.Canny(gray, 40, 130)
    density = float(np.count_nonzero(edges)) / float(edges.size)
    return MIN_ROI_EDGE_DENSITY <= density <= MAX_ROI_EDGE_DENSITY


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
    snapshot = locate_fixed_ui(frame)
    if snapshot is not None:
        region = snapshot.minimap_canvas
        located = (region.bounds, region.confidence, region.method)
        if region.confidence >= MIN_ACCEPTED_SCORE and _valid(frame, region.bounds):
            return located, snapshot

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


def _publish_live_minimap(self) -> bool:
    """Refresh the GUI preview from the current frame on every capture cycle."""
    frame = self.frame
    if frame is None:
        return False
    bounds = (self._minimap_tl, self._minimap_br)
    if not _valid(frame, bounds):
        return False
    (x1, y1), (x2, y2) = bounds
    live = frame[y1:y2, x1:x2]
    if live.size == 0:
        return False
    with self._state_lock:
        self.minimap_sample = live.copy()
        if isinstance(getattr(self, "minimap", None), dict):
            self.minimap["minimap"] = live.copy()
    return True


def _clear_pending(self) -> None:
    self._classic_pending_bounds = None
    self._classic_pending_count = 0
    self._classic_pending_started = 0.0


def _apply_location(self, frame, located, ui_snapshot, reset_tracking: bool) -> bool:
    (top_left, bottom_right), score, method = located
    bounds = (top_left, bottom_right)
    if not _valid(frame, bounds):
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
        self._classic_last_good_bounds = bounds
        self._classic_last_good_frame_shape = frame.shape[:2]
        self._classic_recovery_mode = False
        self._classic_locator_misses = 0
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


def _reuse_last_good_location(self, frame: np.ndarray) -> bool:
    """Restore a previously validated ROI after a transient WGC/watchdog reset."""
    bounds = getattr(self, "_classic_last_good_bounds", None)
    shape = getattr(self, "_classic_last_good_frame_shape", None)
    if bounds is None or shape is None or tuple(shape) != tuple(frame.shape[:2]):
        return False
    if not _crop_looks_alive(frame, bounds):
        return False

    (x1, y1), (x2, y2) = bounds
    sample = frame[y1:y2, x1:x2]
    now = time.monotonic()
    with self._state_lock:
        self._minimap_tl = (x1, y1)
        self._minimap_br = (x2, y2)
        self.minimap_ratio = (x2 - x1) / max(y2 - y1, 1)
        self.minimap_sample = sample.copy()
        self.frame = frame
        self.last_frame_time = now
        self.window["width"] = frame.shape[1]
        self.window["height"] = frame.shape[0]
        self.calibrated = True
        self.last_error = None
        self._classic_recovery_mode = True
        self._classic_last_relocalize = 0.0
        self._classic_locator_misses = 0
        self._reset_tracking()
    print("\n[~] Reusing last known minimap ROI while full localization recovers")
    return True


def install_classic_vision_backend(capture_class) -> None:
    """Install stable minimap calibration with recovery and bounded relocalization."""
    if getattr(capture_class, "_classic_vision_backend_installed", False):
        return

    original_capture_and_track = capture_class._capture_and_track
    original_init = capture_class.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._classic_last_good_bounds = None
        self._classic_last_good_frame_shape = None
        self._classic_last_relocalize = 0.0
        self._classic_last_relock = 0.0
        self._classic_pending_bounds = None
        self._classic_pending_count = 0
        self._classic_pending_started = 0.0
        self._classic_recovery_mode = False
        self._classic_locator_misses = 0

    def calibrate(self) -> bool:
        frame = self.screenshot(delay=0)
        if frame is None:
            return False

        _publish_frame_heartbeat(self, frame)

        result = locate_classic_minimap(frame)
        if result is not None:
            located, ui_snapshot = result
            if _apply_location(self, frame, located, ui_snapshot, reset_tracking=True):
                (top_left, bottom_right), score, method = located
                x1, y1 = top_left
                x2, y2 = bottom_right
                self._classic_last_relocalize = time.monotonic()
                self._classic_last_relock = time.monotonic()
                _clear_pending(self)
                print(
                    f"\n[~] Classic minimap locked using {method} "
                    f"(score {score:.2f}, {x2 - x1}x{y2 - y1}, "
                    f"bounds=({x1},{y1})-({x2},{y2}))"
                )
                return True

        # Most calibration losses in practice are WGC/watchdog hiccups, not the
        # minimap physically moving.  Keep the known crop alive and retry the
        # expensive detector in the background.
        if _reuse_last_good_location(self, frame):
            return True

        self.last_error = (
            "Classic minimap not found safely. Recovery is retrying; keep the "
            "minimap expanded and leave the game window at its normal size."
        )
        return False

    def capture_and_track_with_stable_roi(self) -> bool:
        ok = original_capture_and_track(self)
        if not ok or self.frame is None:
            return ok

        _publish_live_minimap(self)

        now = time.monotonic()
        recovery_mode = bool(getattr(self, "_classic_recovery_mode", False))
        interval = RECOVERY_RETRY_INTERVAL_SECONDS if recovery_mode else RELOCALIZE_INTERVAL_SECONDS
        last = getattr(self, "_classic_last_relocalize", 0.0)
        if now - last < interval:
            return ok
        self._classic_last_relocalize = now

        result = locate_classic_minimap(self.frame)
        if result is None:
            self._classic_locator_misses = getattr(self, "_classic_locator_misses", 0) + 1
            _clear_pending(self)

            # Do not kill WGC/FPS merely because the expensive locator missed a
            # few animated frames.  In recovery mode, however, eventually stop
            # trusting the provisional ROI if the real panel never comes back.
            if recovery_mode and self._classic_locator_misses >= RECOVERY_LOCALIZER_MISS_LIMIT:
                self._classic_recovery_mode = False
                self.last_error = (
                    "Minimap ROI recovery timed out. Keep the minimap expanded; "
                    "Auto Maple will continue searching."
                )
                self.calibrated = False
                self._reset_tracking()
            return ok

        self._classic_locator_misses = 0
        located, ui_snapshot = result
        new_bounds = located[0]
        old_bounds = (self._minimap_tl, self._minimap_br)

        if _bounds_distance(old_bounds, new_bounds) <= RELOCALIZE_CLUSTER_TOLERANCE:
            self._classic_recovery_mode = False
            self._classic_last_good_bounds = old_bounds
            self._classic_last_good_frame_shape = self.frame.shape[:2]
            _clear_pending(self)
            return ok

        if not _plausible_transition(old_bounds, new_bounds):
            _clear_pending(self)
            return ok

        # Immediately after a successful relock, ignore detector wobble.  This is
        # what prevents map animation from repeatedly producing 191/202/261px
        # bottom edges as seen in the classic client.
        last_relock = getattr(self, "_classic_last_relock", 0.0)
        if not recovery_mode and now - last_relock < RELOCALIZE_COOLDOWN_SECONDS:
            _clear_pending(self)
            return ok

        pending = getattr(self, "_classic_pending_bounds", None)
        if pending is None or _bounds_distance(pending, new_bounds) > RELOCALIZE_CLUSTER_TOLERANCE:
            self._classic_pending_bounds = new_bounds
            self._classic_pending_count = 1
            self._classic_pending_started = now
            return ok

        self._classic_pending_count = getattr(self, "_classic_pending_count", 0) + 1
        required = RELOCALIZE_FAST_CONFIRMATIONS if recovery_mode or not self.player_found else RELOCALIZE_CONFIRMATIONS
        pending_started = getattr(self, "_classic_pending_started", now)
        if self._classic_pending_count < required:
            return ok
        if not recovery_mode and self.player_found and now - pending_started < RELOCALIZE_MIN_PENDING_SECONDS:
            return ok

        if _apply_location(self, self.frame, located, ui_snapshot, reset_tracking=True):
            (x1, y1), (x2, y2) = new_bounds
            self._classic_last_relock = now
            print(
                "\n[~] Confirmed minimap layout change; ROI relocked "
                f"to ({x1},{y1})-({x2},{y2})"
            )

        _clear_pending(self)
        return ok

    capture_class.__init__ = patched_init
    capture_class._calibrate_minimap = calibrate
    capture_class._capture_and_track = capture_and_track_with_stable_roi
    capture_class._classic_vision_backend_installed = True
