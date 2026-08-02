"""Stable client-area capture with adaptive foreground/background backends.

The Capture class upstream tracks the outer window rectangle.  Background capture
uses client-area pixels.  These coordinate systems must never overwrite each
other during the upstream window-change comparison, otherwise every frame looks
like a resize and minimap calibration continuously resets.

When the game is foreground, MSS is preferred for maximum FPS.  When the game is
in the background, PrintWindow is preferred so ordinary overlapping windows do
not contaminate the captured image.  PrintWindow support remains dependent on the
game and graphics stack and is usually slower than desktop capture.
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
    """Render only the target window client area into a BGRA image."""
    if not handle or width <= 0 or height <= 0:
        return None

    hwnd_dc = 0
    source_dc = None
    memory_dc = None
    bitmap = None
    try:
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
    """Patch Capture with stable client coordinates and adaptive capture."""
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
        self._outer_window_rect = None

    def patched_refresh_window(self):
        # Upstream compares self.window against GetWindowRect. Restore the last
        # outer rectangle before calling it so a client rectangle is never
        # mistaken for a resize on every frame.
        if self._outer_window_rect is not None:
            self.window = dict(self._outer_window_rect)

        old_handle = int(getattr(self, "_handle", 0))
        previous_client = dict(self._client_rect) if self._client_rect else None
        ok = original_refresh_window(self)
        if not ok:
            self._client_rect = None
            self._outer_window_rect = None
            return False

        # original_refresh_window has now populated the true outer rectangle.
        outer_rect = dict(getattr(self, "window", {}) or {})
        self._outer_window_rect = outer_rect

        client_rect = _client_screen_rect(int(getattr(self, "_handle", 0)))
        if client_rect is None:
            self.window = outer_rect
            return True

        handle_changed = old_handle not in (0, int(getattr(self, "_handle", 0)))
        client_changed = previous_client is not None and client_rect != previous_client
        self._client_rect = dict(client_rect)
        self.window = dict(client_rect)

        # Only a real handle/client-size change invalidates calibration. Merely
        # switching between outer/client coordinate representations does not.
        if handle_changed or client_changed:
            self.calibrated = False
            if hasattr(self, "_reset_tracking"):
                self._reset_tracking()
        return True

    def _mss_client(self, delay: float):
        # original_screenshot reads self.window. At this point it is the client
        # screen rectangle, so MSS and PrintWindow return identical dimensions.
        frame = original_screenshot(self, delay=delay)
        if frame is not None:
            self.capture_backend = "MSS client foreground"
        return frame

    def patched_screenshot(self, delay: float = 1.0):
        now = time.monotonic()
        handle = int(getattr(self, "_handle", 0))
        rect = self._client_rect or self.window
        width = int(rect.get("width", 0))
        height = int(rect.get("height", 0))

        # MSS is much faster and is safe when the game owns the foreground.
        if handle and user32.GetForegroundWindow() == handle:
            return _mss_client(self, delay)

        # In the background, prefer an occlusion-resistant rendered frame.
        if now >= self._background_retry_after:
            frame = _print_client_bgra(handle, width, height)
            if frame is not None:
                self.capture_backend = "PrintWindow client background"
                self._background_failures = 0
                return frame

            self._background_failures += 1
            if self._background_failures >= 3:
                self._background_retry_after = now + 2.0
                self._background_failures = 0

        # If PrintWindow is unsupported, preserve functionality with desktop
        # capture. This fallback may include overlapping windows.
        frame = _mss_client(self, delay)
        if frame is not None:
            self.capture_backend = "MSS client background fallback"
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
