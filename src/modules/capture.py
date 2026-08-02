"""Screen capture, minimap calibration, and player-position tracking.

This module deliberately fails safe: when the game window disappears, is minimized,
or the minimap/player cannot be located reliably, automation is paused and held keys
are released.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

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

GAME_WINDOW_TITLE = "MapleStory"
TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS
WINDOW_RETRY_DELAY = 1.0
CALIBRATION_RETRY_DELAY = 0.5
MINIMAP_MATCH_THRESHOLD = 0.72
PLAYER_MATCH_THRESHOLD = 0.80
PLAYER_LOST_TIMEOUT = 1.5

MINIMAP_TOP_BORDER = 5
MINIMAP_BOTTOM_BORDER = 9


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


MM_TL_TEMPLATE = _load_template("assets/minimap_tl_template.png")
MM_BR_TEMPLATE = _load_template("assets/minimap_br_template.png")
PLAYER_TEMPLATE = _load_template("assets/player_template.png")

MMT_HEIGHT = max(MM_TL_TEMPLATE.shape[0], MM_BR_TEMPLATE.shape[0])
MMT_WIDTH = max(MM_TL_TEMPLATE.shape[1], MM_BR_TEMPLATE.shape[1])
PT_HEIGHT, PT_WIDTH = PLAYER_TEMPLATE.shape


class Capture:
    """Continuously capture the game and publish safe minimap state."""

    def __init__(self):
        config.capture = self

        self.frame: Optional[np.ndarray] = None
        self.minimap = {}
        self.minimap_ratio = 1.0
        self.minimap_sample: Optional[np.ndarray] = None
        self.sct: Optional[mss.mss] = None
        self.window = {"left": 0, "top": 0, "width": 1366, "height": 768}

        self.ready = False
        self.calibrated = False
        self.window_found = False
        self.player_found = False
        self.last_error: Optional[str] = None
        self.last_frame_time = 0.0
        self.last_player_seen = 0.0
        self.frame_id = 0

        self._handle = 0
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._main,
            name="auto-maple-capture",
            daemon=True,
        )

    def start(self):
        print("\n[~] Started video capture")
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        self._pause_for_safety("Capture stopped")

    def _main(self):
        """Wait for the window, calibrate, then track at a bounded frame rate."""

        mss.windows.CAPTUREBLT = 0
        self.ready = True  # The service is ready even if the game has not opened yet.

        try:
            with mss.mss() as sct:
                self.sct = sct
                while not self._stop_event.is_set():
                    if not self._refresh_window():
                        self._pause_for_safety("Waiting for MapleStory window")
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

    def _refresh_window(self) -> bool:
        handle = user32.FindWindowW(None, GAME_WINDOW_TITLE)
        if not handle or not user32.IsWindow(handle) or user32.IsIconic(handle):
            self._handle = 0
            self.window_found = False
            self.calibrated = False
            return False

        rect = wintypes.RECT()
        if not user32.GetWindowRect(handle, ctypes.pointer(rect)):
            self.window_found = False
            self.calibrated = False
            return False

        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width < MMT_WIDTH or height < MMT_HEIGHT:
            self.window_found = False
            self.calibrated = False
            return False

        new_window = {
            "left": max(0, rect.left),
            "top": max(0, rect.top),
            "width": width,
            "height": height,
        }

        # A moved or resized window invalidates the old minimap crop.
        if self._handle and (handle != self._handle or new_window != self.window):
            self.calibrated = False

        self._handle = handle
        self.window = new_window
        self.window_found = True
        return True

    @staticmethod
    def _best_match(frame: np.ndarray, template: np.ndarray):
        if frame is None or frame.size == 0:
            return None
        if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
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
                f"Minimap calibration confidence too low "
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

        print(
            f"\n[~] Minimap calibrated "
            f"(confidence {min(tl_score, br_score):.2f}, {width}x{height})"
        )
        return True

    def _capture_and_track(self) -> bool:
        if not self._refresh_window():
            self._pause_for_safety("MapleStory window lost")
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

        player = utils.multi_match(minimap, PLAYER_TEMPLATE, threshold=PLAYER_MATCH_THRESHOLD)
        now = time.monotonic()
        if player:
            new_position = utils.convert_to_relative(player[0], minimap)
            config.player_pos = new_position
            self.last_player_seen = now
            self.player_found = True
        else:
            self.player_found = False
            if self.last_player_seen and now - self.last_player_seen > PLAYER_LOST_TIMEOUT:
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
                "window_found": self.window_found,
                "frame_id": self.frame_id,
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
