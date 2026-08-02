"""Occlusion-resistant Windows capture backend and bounded FPS tracking.

The upstream capture path uses MSS, which reads desktop pixels and therefore sees
whatever overlaps the game window.  This module first asks the target window to
render itself into a memory bitmap with PrintWindow.  If the game or graphics
stack refuses that request, capture safely falls back to the original MSS path.

PrintWindow support is application-dependent.  It usually handles ordinary
window overlap, but protected or GPU-exclusive surfaces may return a black frame,
and minimized windows are not guaranteed to render.
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
from ctypes import wintypes


user32 = ctypes.windll.user32
PW_RENDERFULLCONTENT = 0x00000002


def _looks_usable(frame: Optional[np.ndarray]) -> bool:
    """Reject empty, nearly black, or effectively constant PrintWindow frames."""
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return False
    sample = frame[:: max(1, frame.shape[0] // 120), :: max(1, frame.shape[1] // 160), :3]
    if sample.size == 0:
        return False
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) > 2.0 and float(gray.std()) > 2.0


def _print_window_bgra(handle: int, width: int, height: int) -> Optional[np.ndarray]:
    """Render HANDLE into a BGRA NumPy image without reading desktop pixels."""
    if not handle or width <= 0 or height <= 0:
        return None

    hwnd_dc = 0
    source_dc = None
    memory_dc = None
    bitmap = None
    try:
        hwnd_dc = user32.GetWindowDC(handle)
        if not hwnd_dc:
            return None

        source_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        memory_dc = source_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(source_dc, width, height)
        memory_dc.SelectObject(bitmap)

        rendered = user32.PrintWindow(handle, memory_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
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
    """Patch Capture with PrintWindow-first capture and rolling FPS metrics."""
    if getattr(capture_class, "_background_capture_patch_installed", False):
        return

    original_init = capture_class.__init__
    original_screenshot = capture_class.screenshot
    original_capture_and_track = capture_class._capture_and_track

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.fps = 0.0
        self.capture_backend = "initializing"
        self._frame_times = deque(maxlen=120)
        self._background_failures = 0
        self._background_retry_after = 0.0

    def patched_screenshot(self, delay: float = 1.0):
        now = time.monotonic()
        width = int(self.window.get("width", 0))
        height = int(self.window.get("height", 0))

        # Avoid repeatedly calling a backend that the client has just rejected.
        if now >= self._background_retry_after:
            frame = _print_window_bgra(int(getattr(self, "_handle", 0)), width, height)
            if frame is not None:
                self.capture_backend = "PrintWindow"
                self._background_failures = 0
                return frame

            self._background_failures += 1
            # Back off briefly after repeated failures, while preserving MSS capture.
            if self._background_failures >= 3:
                self._background_retry_after = now + 2.0
                self._background_failures = 0

        frame = original_screenshot(self, delay=delay)
        if frame is not None:
            self.capture_backend = "MSS desktop fallback"
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
                if elapsed > 0:
                    self.fps = (len(self._frame_times) - 1) / elapsed
            else:
                self.fps = 0.0
        return ok

    capture_class.__init__ = patched_init
    capture_class.screenshot = patched_screenshot
    capture_class._capture_and_track = patched_capture_and_track
    capture_class._background_capture_patch_installed = True
