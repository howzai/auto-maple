"""Classic-client vision backend with optional YOLO and stable UI localization."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.modules.classic_minimap import find_classic_minimap
from src.modules.manual_minimap_roi import load_search_region
from src.vision.fixed_ui import FixedUiSnapshot, locate_fixed_ui


Bounds = Tuple[Tuple[int, int], Tuple[int, int]]
Located = Tuple[Bounds, float, str]

_MODEL = None
_MODEL_ATTEMPTED = False

RELOCALIZE_INTERVAL_SECONDS = 2.0
RELOCALIZE_CONFIRMATIONS = 3
RELOCALIZE_CLUSTER_TOLERANCE = 8
MAX_SINGLE_STEP_PIXELS = 28
MAX_SIZE_CHANGE_RATIO = 0.22

# Fast recovery is intentionally aggressive only after startup has already locked.
MAP_CHANGE_PLAYER_LOST_SECONDS = 0.45
MAP_CHANGE_SCAN_INTERVAL_SECONDS = 0.12
MAP_CHANGE_CONFIRMATIONS = 2
MAP_CHANGE_TOP_LEFT_TOLERANCE = 34
MAP_CHANGE_MAX_PENDING_SECONDS = 1.5

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
        and x1 <= max(100, int(width * 0.12))
        and y1 <= max(110, int(height * 0.16))
        and 90 <= crop_width <= max(460, int(width * 0.38))
        and 30 <= crop_height <= max(360, int(height * 0.46))
        and 0.40 <= crop_width / max(crop_height, 1) <= 7.5
    )


def _translate(bounds: Bounds, ox: int, oy: int) -> Bounds:
    return (
        (bounds[0][0] + ox, bounds[0][1] + oy),
        (bounds[1][0] + ox, bounds[1][1] + oy),
    )


def _find_in_manual_region(frame: np.ndarray) -> Optional[Located]:
    """Fast path: search only the user-selected F11 rectangle."""
    region = load_search_region(frame)
    if region is None:
        return None
    (x1, y1), (x2, y2) = region
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # The legacy contour detector is excellent once the search area has already
    # been constrained by the user.  Translate its local coordinates back to the
    # WGC game frame.
    legacy = find_classic_minimap(crop)
    if legacy is not None:
        local_bounds, score, method = legacy
        bounds = _translate(local_bounds, x1, y1)
        if _valid(frame, bounds):
            return bounds, float(score), f"manual-{method}"

    # If the user drew a tight box and contour detection misses transiently,
    # derive a conservative canvas from the selected panel.  This is only a
    # temporary fallback; normal relocalization will replace it when available.
    rw, rh = x2 - x1, y2 - y1
    if rw >= 150 and rh >= 110:
        left = x1 + max(3, int(rw * 0.02))
        right = x2 - max(3, int(rw * 0.03))
        top = y1 + max(42, int(rh * 0.34))
        bottom = y2 - max(8, int(rh * 0.08))
        bounds = ((left, top), (right, bottom))
        if _valid(frame, bounds):
            return bounds, 0.50, "manual-box-fallback"
    return None


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
    """Locate the minimap, preferring the persistent F11 user search region."""
    manual = _find_in_manual_region(frame)
    if manual is not None:
        return manual, None

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


def _top_left_distance(a: Bounds, b: Bounds) -> float:
    return math.hypot(a[0][0] - b[0][0], a[0][1] - b[0][1])


def _plausible_transition(old_bounds: Bounds, new_bounds: Bounds) -> bool:
    old_width, old_height = _size(old_bounds)
    new_width, new_height = _size(new_bounds)
    if old_width <= 0 or old_height <= 0:
        return True

    top_left_shift = _top_left_distance(old_bounds, new_bounds)
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
            1.0 / ANCHORED_RESIZE_MAX_WIDTH_RATIO <= width_scale <= ANCHORED_RESIZE_MAX_WIDTH_RATIO
            and 1.0 / ANCHORED_RESIZE_MAX_HEIGHT_RATIO <= height_scale <= ANCHORED_RESIZE_MAX_HEIGHT_RATIO
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


def _clear_pending(self) -> None:
    self._classic_pending_bounds = None
    self._classic_pending_count = 0
    self._classic_pending_started = 0.0


def _apply_location(self, frame, located, ui_snapshot, reset_tracking: bool) -> bool:
    (top_left, bottom_right), _score, method = located
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
        self._classic_ever_locked = True
        self._classic_last_good_bounds = (top_left, bottom_right)
        if reset_tracking:
            self._reset_tracking()
    return True


def _confirm_candidate(self, new_bounds: Bounds, confirmations: int) -> bool:
    now = time.monotonic()
    pending = getattr(self, "_classic_pending_bounds", None)
    started = getattr(self, "_classic_pending_started", 0.0)

    if pending is None or _bounds_distance(pending, new_bounds) > RELOCALIZE_CLUSTER_TOLERANCE:
        self._classic_pending_bounds = new_bounds
        self._classic_pending_count = 1
        self._classic_pending_started = now
        return confirmations <= 1

    if started and now - started > MAP_CHANGE_MAX_PENDING_SECONDS:
        self._classic_pending_bounds = new_bounds
        self._classic_pending_count = 1
        self._classic_pending_started = now
        return False

    self._classic_pending_count = getattr(self, "_classic_pending_count", 0) + 1
    return self._classic_pending_count >= confirmations


def _fast_map_change_recovery(self, frame: np.ndarray) -> bool:
    if not getattr(self, "_classic_ever_locked", False):
        return False

    now = time.monotonic()
    last_scan = getattr(self, "_classic_last_recovery_scan", 0.0)
    if now - last_scan < MAP_CHANGE_SCAN_INTERVAL_SECONDS:
        return False
    self._classic_last_recovery_scan = now

    result = locate_classic_minimap(frame)
    if result is None:
        return False

    located, ui_snapshot = result
    new_bounds = located[0]
    old_bounds = (self._minimap_tl, self._minimap_br)

    if _bounds_distance(old_bounds, new_bounds) <= RELOCALIZE_CLUSTER_TOLERANCE:
        _clear_pending(self)
        return False

    # With an F11 search region, a map resize is trusted primarily because the
    # user has already constrained the only place a minimap is allowed to exist.
    manual_active = load_search_region(frame) is not None
    if not manual_active and _top_left_distance(old_bounds, new_bounds) > MAP_CHANGE_TOP_LEFT_TOLERANCE:
        _clear_pending(self)
        return False
    if not _confirm_candidate(self, new_bounds, MAP_CHANGE_CONFIRMATIONS):
        return False

    if _apply_location(self, frame, located, ui_snapshot, reset_tracking=True):
        (x1, y1), (x2, y2) = new_bounds
        self._classic_last_relocalize = now
        _clear_pending(self)
        print(f"\n[~] Map change detected; minimap ROI recovered to ({x1},{y1})-({x2},{y2})")
        return True
    return False


def install_classic_vision_backend(capture_class) -> None:
    """Install stable minimap calibration plus post-start map-change recovery."""
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
                "Classic minimap not found safely. Press F11 and draw a box around "
                "the whole minimap panel, or keep the minimap expanded."
            )
            return False

        located, ui_snapshot = result
        if not _apply_location(self, frame, located, ui_snapshot, reset_tracking=True):
            return False

        (top_left, bottom_right), score, method = located
        x1, y1 = top_left
        x2, y2 = bottom_right
        self._classic_last_relocalize = time.monotonic()
        self._classic_last_recovery_scan = 0.0
        self._classic_player_missing_since = 0.0
        _clear_pending(self)
        print(
            f"\n[~] Classic minimap locked using {method} "
            f"(score {score:.2f}, {x2 - x1}x{y2 - y1}, bounds=({x1},{y1})-({x2},{y2}))"
        )
        return True

    def capture_and_track_with_stable_roi(self) -> bool:
        ok = original_capture_and_track(self)

        if not ok:
            if getattr(self, "_classic_ever_locked", False):
                frame = self.screenshot(delay=0)
                if frame is not None:
                    _publish_frame_heartbeat(self, frame)
                    if _fast_map_change_recovery(self, frame):
                        return True
            return False

        if self.frame is None:
            return ok

        _publish_live_minimap(self)
        now = time.monotonic()

        if getattr(self, "player_found", False):
            self._classic_player_missing_since = 0.0
        else:
            missing_since = getattr(self, "_classic_player_missing_since", 0.0)
            if not missing_since:
                self._classic_player_missing_since = now
            elif now - missing_since >= MAP_CHANGE_PLAYER_LOST_SECONDS:
                if _fast_map_change_recovery(self, self.frame):
                    self._classic_player_missing_since = 0.0
                    return True

        last = getattr(self, "_classic_last_relocalize", 0.0)
        if now - last < RELOCALIZE_INTERVAL_SECONDS:
            return ok
        self._classic_last_relocalize = now

        result = locate_classic_minimap(self.frame)
        if result is None:
            _clear_pending(self)
            return ok

        located, ui_snapshot = result
        new_bounds = located[0]
        old_bounds = (self._minimap_tl, self._minimap_br)

        if _bounds_distance(old_bounds, new_bounds) <= RELOCALIZE_CLUSTER_TOLERANCE:
            _clear_pending(self)
            return ok
        if not _plausible_transition(old_bounds, new_bounds) and load_search_region(self.frame) is None:
            _clear_pending(self)
            return ok
        if not _confirm_candidate(self, new_bounds, RELOCALIZE_CONFIRMATIONS):
            return ok

        if _apply_location(self, self.frame, located, ui_snapshot, reset_tracking=True):
            (x1, y1), (x2, y2) = new_bounds
            print(f"\n[~] Confirmed minimap layout change; ROI relocked to ({x1},{y1})-({x2},{y2})")

        _clear_pending(self)
        return ok

    capture_class._calibrate_minimap = calibrate
    capture_class._capture_and_track = capture_and_track_with_stable_roi
    capture_class._classic_vision_backend_installed = True
