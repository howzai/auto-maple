"""Screen capture, minimap calibration, and player-position tracking.

The capture service fails safe. When the game window disappears, is minimized, the
capture stream stalls, or the minimap/player cannot be located reliably, automation
is paused and all held keys are released.
"""

from __future__ import annotations

import ctypes
import math
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import mss
import mss.windows
import numpy as np
from ctypes import wintypes

from src.common import config, utils
from src.common.vkeys import release_all


user32 = ctypes.windll.user32
try:
    user32.SetProcessDPIAware()
except Exception:
    pass

# Exact titles are checked first. A conservative partial-title fallback is then
# used for regional clients whose title contains MapleStory / 楓之谷.
GAME_WINDOW_TITLES = (
    "MapleStory",
    "新楓之谷",
    "新楓之谷：經典版",
    "新楓之谷:經典版",
)
GAME_WINDOW_TITLE_KEYWORDS = ("maplestory", "楓之谷")

TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS
WINDOW_RETRY_DELAY = 1.0
CALIBRATION_RETRY_DELAY = 0.5
MINIMAP_MATCH_THRESHOLD = 0.72
PLAYER_MATCH_THRESHOLD = 0.80
PLAYER_LOST_TIMEOUT = 1.5
CAPTURE_STALL_TIMEOUT = 2.5
WATCHDOG_INTERVAL = 0.25

# Player tracking parameters. Coordinates are normalized minimap coordinates.
POSITION_EMA_ALPHA = 0.45
MAX_NORMAL_JUMP = 0.18
TELEPORT_CONFIRM_FRAMES = 3
TELEPORT_CLUSTER_RADIUS = 0.08

MINIMAP_TOP_BORDER = 5
MINIMAP_BOTTOM_BORDER = 9

Point = Tuple[float, float]


def _project_root() -> Path:
    """Return a stable project root in source and PyInstaller environments."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def _load_template(relative_path: str) -> np.ndarray:
    path = _project_root() / relative_path
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Required image template could not be loaded: {path}")
    return image


def _window_title(handle: int) -> str:
    length = user32.GetWindowTextLengthW(handle)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, len(buffer))
    return buffer.value.strip()


def _find_game_window() -> Tuple[int, str]:
    """Find a visible MapleStory client window across regional title variants."""
    for title in GAME_WINDOW_TITLES:
        handle = user32.FindWindowW(None, title)
        if handle and user32.IsWindow(handle):
            return int(handle), title

    matches = []
    enum_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @enum_proc_type
    def enum_proc(handle, _):
        if not user32.IsWindowVisible(handle):
            return True
        title = _window_title(handle)
        normalized = title.casefold()
        if title and any(keyword in normalized for keyword in GAME_WINDOW_TITLE_KEYWORDS):
            matches.append((int(handle), title))
        return True

    user32.EnumWindows(enum_proc, 0)
    if not matches:
        return 0, ""

    # Prefer the largest matching visible window, which avoids launchers or tiny
    # helper windows that may also contain MapleStory in their title.
    def area(item):
        rect = wintypes.RECT()
        if not user32.GetWindowRect(item[0], ctypes.pointer(rect)):
            return 0
        return max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)

    return max(matches, key=area)


MM_TL_TEMPLATE = _load_template("assets/minimap_tl_template.png")
MM_BR_TEMPLATE = _load_template("assets/minimap_br_template.png")
PLAYER_TEMPLATE = _load_template("assets/player_template.png")

MMT_HEIGHT = max(MM_TL_TEMPLATE.shape[0], MM_BR_TEMPLATE.shape[0])
MMT_WIDTH = max(MM_TL_TEMPLATE.shape[1], MM_BR_TEMPLATE.shape[1])
PT_HEIGHT, PT_WIDTH = PLAYER_TEMPLATE.shape


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Capture:
    """Continuously capture the game and publish safe minimap state."""

    def __init__(self):
        config.capture = self

        self.frame: Optional[np.ndarray] = None
        self.minimap: Dict[str, object] = {}
        self.minimap_ratio = 1.0
        self.minimap_sample: Optional[np.ndarray] = None
        self.sct: Optional[mss.mss] = None
        self.window = {"left": 0, "top": 0, "width": 1366, "height": 768}
        self.window_title = ""

        self.ready = False
        self.calibrated = False
        self.window_found = False
        self.player_found = False
        self.last_error: Optional[str] = None
        self.last_frame_time = 0.0
        self.last_player_seen = 0.0
        self.frame_id = 0
        self.player_confidence = 0.0
        self.rejected_player_jumps = 0

        self._handle = 0
        self._filtered_position: Optional[Point] = None
        self._pending_jump: Optional[Point] = None
        self._pending_jump_frames = 0
        self._minimap_tl = (0, 0)
        self._minimap_br = (0, 0)
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._main,
            name="auto-maple-capture",
            daemon=True,
        )
        self.watchdog_thread = threading.Thread(
            target=self._watchdog,
            name="auto-maple-capture-watchdog",
            daemon=True,
        )

    def start(self):
        print("\n[~] Started video capture")
        self.thread.start()
        self.watchdog_thread.start()

    def stop(self):
        self._stop_event.set()
        self._pause_for_safety("Capture stopped")

    def health_snapshot(self) -> Dict[str, object]:
        """Return an immutable diagnostic snapshot for GUI/logging code."""
        with self._state_lock:
            age = None
            if self.last_frame_time:
                age = max(0.0, time.monotonic() - self.last_frame_time)
            return {
                "ready": self.ready,
                "thread_alive": self.thread.is_alive(),
                "watchdog_alive": self.watchdog_thread.is_alive(),
                "window_found": self.window_found,
                "window_title": self.window_title,
                "calibrated": self.calibrated,
                "player_found": self.player_found,
                "player_confidence": self.player_confidence,
                "frame_id": self.frame_id,
                "last_frame_age": age,
                "rejected_player_jumps": self.rejected_player_jumps,
                "last_error": self.last_error,
            }

    def _main(self):
        """Wait for the window, calibrate, then track at a bounded frame rate."""
        mss.windows.CAPTUREBLT = 0
        self.ready = True

        try:
            with mss.mss() as sct:
                self.sct = sct
                while not self._stop_event.is_set():
                    if not self._refresh_window():
                        self._pause_for_safety("Waiting for game window")
                        time.sleep(WINDOW_RETRY_DELAY)
                        continue

                    if not self.calibrated:
                        if not self._calibrate_minimap():
                            time.sleep(CALIBRATION_RETRY_DELAY)
                            continue

                    started = time.perf_counter()
                    if not self._capture_and_track():
                        self.calibrated = False

                    remaining = FRAME_INTERVAL - (time.perf_counter() - started)
                    if remaining > 0:
                        time.sleep(remaining)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._pause_for_safety("Capture thread crashed")
            print("\n[!] Capture thread stopped unexpectedly:")
            traceback.print_exc()
        finally:
            self.sct = None
            self.window_found = False
            self.calibrated = False

    def _watchdog(self):
        """Pause automation when the capture worker dies or stops producing frames."""
        while not self._stop_event.is_set():
            time.sleep(WATCHDOG_INTERVAL)
            if not self.ready:
                continue

            if not self.thread.is_alive():
                self.last_error = self.last_error or "Capture worker is not running"
                self._pause_for_safety("Capture worker stopped")
                return

            if not self.window_found or not self.calibrated or not self.last_frame_time:
                continue

            frame_age = time.monotonic() - self.last_frame_time
            if frame_age > CAPTURE_STALL_TIMEOUT:
                self.last_error = f"Capture stalled for {frame_age:.2f} seconds"
                self.calibrated = False
                self._pause_for_safety("Capture stream stalled")

    def _reset_tracking(self):
        with self._state_lock:
            self.player_found = False
            self.player_confidence = 0.0
            self._filtered_position = None
            self._pending_jump = None
            self._pending_jump_frames = 0
            self.last_player_seen = 0.0

    def _refresh_window(self) -> bool:
        handle, title = _find_game_window()
        if not handle or not user32.IsWindow(handle) or user32.IsIconic(handle):
            self._handle = 0
            self.window_title = ""
            self.window_found = False
            self.calibrated = False
            self._reset_tracking()
            return False

        rect = wintypes.RECT()
        if not user32.GetWindowRect(handle, ctypes.pointer(rect)):
            self.window_found = False
            self.calibrated = False
            self._reset_tracking()
            return False

        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width < MMT_WIDTH or height < MMT_HEIGHT:
            self.last_error = f"Matched game window is too small: {width}x{height}"
            self.window_found = False
            self.calibrated = False
            self._reset_tracking()
            return False

        new_window = {
            "left": max(0, rect.left),
            "top": max(0, rect.top),
            "width": width,
            "height": height,
        }

        changed = handle != self._handle or new_window != self.window
        newly_found = handle != self._handle
        if self._handle and changed:
            self.calibrated = False
            self._reset_tracking()

        self._handle = handle
        self.window_title = title
        self.window = new_window
        self.window_found = True
        if newly_found:
            self.last_error = None
            print(f"\n[~] Game window found: '{title}' ({width}x{height})")
        return True

    @staticmethod
    def _best_match(frame: np.ndarray, template: np.ndarray):
        if frame is None or frame.size == 0:
            return None
        if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
            return None

        conversion = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(frame, conversion)
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, top_left = cv2.minMaxLoc(result)
        h, w = template.shape
        return float(score), top_left, (top_left[0] + w, top_left[1] + h)

    def _calibrate_minimap(self) -> bool:
        frame = self.screenshot(delay=0)
        if frame is None:
            return False

        tl_match = self._best_match(frame, MM_TL_TEMPLATE)
        br_match = self._best_match(frame, MM_BR_TEMPLATE)
        if not tl_match or not br_match:
            return False

        tl_score, tl, _ = tl_match
        br_score, _, br = br_match
        if tl_score < MINIMAP_MATCH_THRESHOLD or br_score < MINIMAP_MATCH_THRESHOLD:
            self.last_error = (
                "Minimap calibration confidence too low "
                f"(top-left={tl_score:.2f}, bottom-right={br_score:.2f})"
            )
            return False

        mm_tl = (tl[0] + MINIMAP_BOTTOM_BORDER, tl[1] + MINIMAP_TOP_BORDER)
        mm_br = (
            max(mm_tl[0] + PT_WIDTH, br[0] - MINIMAP_BOTTOM_BORDER),
            max(mm_tl[1] + PT_HEIGHT, br[1] - MINIMAP_BOTTOM_BORDER),
        )

        if mm_br[0] > frame.shape[1] or mm_br[1] > frame.shape[0]:
            self.last_error = "Detected minimap bounds exceed captured frame"
            return False

        width = mm_br[0] - mm_tl[0]
        height = mm_br[1] - mm_tl[1]
        if width <= 0 or height <= 0:
            self.last_error = "Detected minimap has invalid dimensions"
            return False

        sample = frame[mm_tl[1]:mm_br[1], mm_tl[0]:mm_br[0]]
        if sample.size == 0:
            return False

        with self._state_lock:
            self._minimap_tl = mm_tl
            self._minimap_br = mm_br
            self.minimap_ratio = width / height
            self.minimap_sample = sample.copy()
            self.frame = frame
            self.calibrated = True
            self.last_error = None
            self._reset_tracking()

        print(
            f"\n[~] Minimap calibrated "
            f"(confidence {min(tl_score, br_score):.2f}, {width}x{height})"
        )
        return True

    def _player_candidates(self, minimap: np.ndarray) -> Sequence[Tuple[Point, float]]:
        """Return normalized player candidates with template confidence scores."""
        conversion = cv2.COLOR_BGRA2GRAY if minimap.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(minimap, conversion)
        if PLAYER_TEMPLATE.shape[0] > gray.shape[0] or PLAYER_TEMPLATE.shape[1] > gray.shape[1]:
            return []

        result = cv2.matchTemplate(gray, PLAYER_TEMPLATE, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= PLAYER_MATCH_THRESHOLD)
        candidates = []
        for x, y in zip(xs, ys):
            center = (
                int(round(x + PLAYER_TEMPLATE.shape[1] / 2)),
                int(round(y + PLAYER_TEMPLATE.shape[0] / 2)),
            )
            relative = utils.convert_to_relative(center, minimap)
            candidates.append((relative, float(result[y, x])))

        candidates.sort(key=lambda item: item[1], reverse=True)
        deduplicated = []
        for point, score in candidates:
            if all(_distance(point, existing[0]) > 0.025 for existing in deduplicated):
                deduplicated.append((point, score))
        return deduplicated

    def _select_player_candidate(
        self, candidates: Sequence[Tuple[Point, float]]
    ) -> Optional[Tuple[Point, float]]:
        if not candidates:
            return None
        if self._filtered_position is None:
            return max(candidates, key=lambda item: item[1])

        return min(
            candidates,
            key=lambda item: _distance(item[0], self._filtered_position) - item[1] * 0.02,
        )

    def _accept_position(self, position: Point) -> Optional[Point]:
        """Smooth normal motion and require confirmation for large coordinate jumps."""
        if self._filtered_position is None:
            self._filtered_position = position
            return position

        jump = _distance(position, self._filtered_position)
        if jump <= MAX_NORMAL_JUMP:
            self._pending_jump = None
            self._pending_jump_frames = 0
            alpha = POSITION_EMA_ALPHA
            filtered = (
                alpha * position[0] + (1.0 - alpha) * self._filtered_position[0],
                alpha * position[1] + (1.0 - alpha) * self._filtered_position[1],
            )
            self._filtered_position = filtered
            return filtered

        if self._pending_jump is None or _distance(position, self._pending_jump) > TELEPORT_CLUSTER_RADIUS:
            self._pending_jump = position
            self._pending_jump_frames = 1
        else:
            self._pending_jump = (
                (self._pending_jump[0] + position[0]) / 2,
                (self._pending_jump[1] + position[1]) / 2,
            )
            self._pending_jump_frames += 1

        if self._pending_jump_frames >= TELEPORT_CONFIRM_FRAMES:
            self._filtered_position = self._pending_jump
            accepted = self._filtered_position
            self._pending_jump = None
            self._pending_jump_frames = 0
            return accepted

        self.rejected_player_jumps += 1
        return None

    def _capture_and_track(self) -> bool:
        if not self._refresh_window():
            self._pause_for_safety("Game window lost")
            return False

        frame = self.screenshot(delay=0)
        if frame is None:
            return False

        x1, y1 = self._minimap_tl
        x2, y2 = self._minimap_br
        if y2 > frame.shape[0] or x2 > frame.shape[1]:
            return False

        minimap = frame[y1:y2, x1:x2]
        if minimap.size == 0:
            return False

        now = time.monotonic()
        candidates = self._player_candidates(minimap)
        selected = self._select_player_candidate(candidates)
        accepted_position = None

        if selected is not None:
            raw_position, confidence = selected
            accepted_position = self._accept_position(raw_position)
            self.player_confidence = confidence

        if accepted_position is not None:
            config.player_pos = accepted_position
            self.last_player_seen = now
            self.player_found = True
        elif selected is not None and self._filtered_position is not None:
            self.player_found = False
        else:
            self.player_found = False
            self.player_confidence = 0.0

        if not self.player_found and self.last_player_seen and now - self.last_player_seen > PLAYER_LOST_TIMEOUT:
            self._pause_for_safety("Player marker lost")

        with self._state_lock:
            self.frame = frame
            self.frame_id += 1
            self.last_frame_time = now
            self.minimap = {
                "minimap": minimap.copy(),
                "rune_active": bool(getattr(config.bot, "rune_active", False)),
                "rune_pos": getattr(config.bot, "rune_pos", (0, 0)),
                "path": list(config.path),
                "player_pos": config.player_pos,
                "player_found": self.player_found,
                "player_confidence": self.player_confidence,
                "player_candidates": len(candidates),
                "window_found": self.window_found,
                "window_title": self.window_title,
                "calibrated": self.calibrated,
                "frame_id": self.frame_id,
                "rejected_player_jumps": self.rejected_player_jumps,
            }
        return True

    def _pause_for_safety(self, reason: str):
        was_enabled = bool(config.enabled)
        config.enabled = False
        release_all()
        if was_enabled:
            print(f"\n[!] Auto Maple paused for safety: {reason}")

    def screenshot(self, delay: float = 1.0) -> Optional[np.ndarray]:
        if self.sct is None or not self.window_found:
            return None
        try:
            return np.asarray(self.sct.grab(self.window))
        except (mss.exception.ScreenShotError, ValueError) as exc:
            self.last_error = f"Screenshot failed: {exc}"
            if delay:
                print(f"\n[!] Error while taking screenshot; retrying in {delay:g} second(s)")
                time.sleep(delay)
            return None
