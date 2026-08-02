"""Occlusion-resistant client-area capture and bounded FPS tracking.

Both PrintWindow and MSS are kept in the same client-area coordinate system. This
avoids calibration drift caused by mixing whole-window coordinates (title bar and
borders included) with client-only rendered pixels.
"""

from __future__ import annotations

import ctypes
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import win32gui
import win32ui


user32 = ctypes.windll.user32
PW_CLIENTONLY = 0x00000001
PW_RENDERFULLCONTENT = 0x00000002


def _looks_usable(frame: Optional[np.ndarray]) -> bool:
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return False
    sample = frame[
        :: max(1, frame.shape[0] // 120),
        :: max(1, frame.shape[1] // 160),
        :3,
    ]
    if sample.size == 0:
        return False
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) > 2.0 and float(gray.std()) > 2.0


def _client_screen_rect(handle: int):
    """Return client-area bounds in screen coordinates."""
    if not handle or not user32.IsWindow(handle):
        return None
    try:
        left, top, right, bottom = win32gui.GetClientRect(handle)
        screen_left, screen_top = win32gui.ClientToScreen(handle, (left, top))
        screen_right, screen_bottom = win32gui.ClientToScreen(handle, (right, bottom))
        width = max(0, screen_right - screen_left)
        height = max(0, screen_bottom - screen_top)
        if width <= 0 or height <= 0:
            return None
        return {
            "left": int(screen_left),
            "top": int(screen_top),
            "width": int(width),
            "height": int(height),
        }
    except Exception:
        return None


def _print_client_bgra(handle: int, width: int, height: int) -> Optional[np.ndarray]:
    """Render only the target window's client area into a BGRA image."""
    if not handle or width <= 0 or height <= 0:
        return None

    hwnd_dc = 0
    source_dc = None
    memory_dc = None
    bitmap = None
    try:
        # GetDC targets the client area; GetWindowDC would include non-client chrome.
        hwnd_dc = user32.GetDC(handle)
        if not hwnd_dc:
            return None

        source_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        memory_dc = source_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(source_dc, width, height)
        memory_dc.SelectObject(bitmap)

        flags = PW_CLIENTONLY | PW_RENDERFULLCONTENT
        rendered = user32.PrintWindow(handle, memory_dc.GetSafeHdc(), flags)
        if not rendered:
            return None

        raw = bitmap.GetBitmapBits(True)
        frame = np.frombuffer(raw, dtype=np.uint8)
        expected = width * height * 4
        if frame.size != expected:
            return None
        frame = frame.reshape((height, width, 4)).copy()
        return frame if _looks_usable(frame) else None
    except Exception:
        return None
    finally:
        if memory_dc is not None:
            try:
                memory_dc.DeleteDC()
            except Exception:
                pass
        if source_dc is not None:
            try:
                source_dc.DeleteDC()
            except Exception:
                pass
        if bitmap is not None:
            try:
                win32gui.DeleteObject(bitmap.GetHandle())
            except Exception:
                pass
        if hwnd_dc:
            try:
                user32.ReleaseDC(handle, hwnd_dc)
            except Exception:
                pass


def install_background_capture(capture_class) -> None:
    """Patch Capture with aligned client capture and rolling FPS metrics."""
    if getattr(capture_class, "_background_capture_patch_installed", False):
        return

    original_init = capture_class.__init__
    original_refresh_window = capture_class._refresh_window
    original_screenshot = capture_class.screenshot
    original_capture_and_track = capture_class._capture_and_track

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.fps = 0.0
        self.capture_backend = "initializing"
        self._frame_times = deque(maxlen=120)
        self._background_failures = 0
        self._background_retry_after = 0.0
        self._client_rect = None

    def patched_refresh_window(self):
        previous_rect = dict(getattr(self, "window", {}) or {})
        ok = original_refresh_window(self)
        if not ok:
            self._client_rect = None
            return False

        client_rect = _client_screen_rect(int(getattr(self, "_handle", 0)))
        if client_rect is None:
            return True

        # Use one coordinate space for calibration, tracking, PrintWindow and MSS.
        changed = client_rect != self._client_rect
        self._client_rect = dict(client_rect)
        self.window = dict(client_rect)
        if changed and previous_rect != client_rect:
            self.calibrated = False
            if hasattr(self, "_reset_tracking"):
                self._reset_tracking()
        return True

    def patched_screenshot(self, delay: float = 1.0):
        now = time.monotonic()
        rect = self._client_rect or self.window
        width = int(rect.get("width", 0))
        height = int(rect.get("height", 0))

        if now >= self._background_retry_after:
            frame = _print_client_bgra(
                int(getattr(self, "_handle", 0)),
                width,
                height,
            )
            if frame is not None:
                self.capture_backend = "PrintWindow client"
                self._background_failures = 0
                return frame

            self._background_failures += 1
            if self._background_failures >= 3:
                self._background_retry_after = now + 2.0
                self._background_failures = 0

        # original_screenshot reads self.window, now also client-area aligned.
        frame = original_screenshot(self, delay=delay)
        if frame is not None:
            self.capture_backend = "MSS client fallback"
        return frame

    def patched_capture_and_track(self):
        ok = original_capture_and_track(self)
        if ok:
            now = time.monotonic()
            self._frame_times.append(now)
            while self._frame_times and now - self._frame_times[0] > 1.5:
                self._frame_times.popleft()
            if len(self._frame_times) >= 2:
                elapsed = self._frame_times[-1] - self._frame_times[0]
                self.fps = ((len(self._frame_times) - 1) / elapsed) if elapsed > 0 else 0.0
            else:
                self.fps = 0.0
        return ok

    capture_class.__init__ = patched_init
    capture_class._refresh_window = patched_refresh_window
    capture_class.screenshot = patched_screenshot
    capture_class._capture_and_track = patched_capture_and_track
    capture_class._background_capture_patch_installed = True
