"""Robust MapleStory foreground-window guard.

Accepts the captured game window directly, another top-level window from the same
process, or a foreground window whose title clearly identifies MapleStory.  This
keeps patrol input blocked for browsers, CMD, Desktop, and unrelated apps while
avoiding false pauses caused by classic-client HWND changes.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from src.common import config
from src.modules.capture import GAME_WINDOW_TITLE_KEYWORDS, GAME_WINDOW_TITLES


user32 = ctypes.windll.user32


def _window_title(handle: int) -> str:
    if not handle or not user32.IsWindow(handle):
        return ""
    length = user32.GetWindowTextLengthW(handle)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, len(buffer))
    return buffer.value.strip()


def _window_pid(handle: int) -> int:
    if not handle or not user32.IsWindow(handle):
        return 0
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
    return int(pid.value)


def _title_is_maple(title: str) -> bool:
    if not title:
        return False
    normalized = title.casefold()
    if title in GAME_WINDOW_TITLES:
        return True
    return any(keyword.casefold() in normalized for keyword in GAME_WINDOW_TITLE_KEYWORDS)


def install_game_focus_guard(patrol_controller_class) -> None:
    if getattr(patrol_controller_class, "_game_focus_guard_installed", False):
        return

    def foreground_is_game() -> bool:
        capture = getattr(config, "capture", None)
        if capture is None:
            return False

        captured_handle = int(getattr(capture, "_handle", 0) or 0)
        foreground = int(user32.GetForegroundWindow() or 0)
        if not captured_handle or not foreground:
            return False
        if not user32.IsWindow(captured_handle) or not user32.IsWindow(foreground):
            return False

        # Fast path: exact same top-level window.
        if foreground == captured_handle:
            return True

        # Classic MapleStory can recreate/swap its top-level HWND while remaining
        # the same process.  Treat another foreground window from that process as
        # safe game focus.
        captured_pid = _window_pid(captured_handle)
        foreground_pid = _window_pid(foreground)
        if captured_pid and foreground_pid and captured_pid == foreground_pid:
            return True

        # Final conservative fallback for the classic client: the foreground title
        # itself must clearly identify MapleStory.  Unrelated CMD/browser/Desktop
        # windows therefore remain blocked even if the stored capture HWND is stale.
        foreground_title = _window_title(foreground)
        return _title_is_maple(foreground_title)

    patrol_controller_class._foreground_is_game = staticmethod(foreground_is_game)
    patrol_controller_class._game_focus_guard_installed = True
