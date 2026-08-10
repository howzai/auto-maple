"""Robust hotkey polling and safe MapleStory focus checks for patrol input."""

from __future__ import annotations

import ctypes


user32 = ctypes.windll.user32

VK_MAP = {
    "insert": 0x2D,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
}

GAME_TITLE_KEYWORDS = ("maplestory", "楓之谷")


def _window_process_id(hwnd: int) -> int:
    if not hwnd or not user32.IsWindow(hwnd):
        return 0
    pid = ctypes.c_ulong(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _window_title(hwnd: int) -> str:
    if not hwnd or not user32.IsWindow(hwnd):
        return ""
    length = int(user32.GetWindowTextLengthW(hwnd) or 0)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value.strip()


def _title_is_maple(title: str) -> bool:
    normalized = (title or "").casefold()
    return bool(normalized and any(keyword.casefold() in normalized for keyword in GAME_TITLE_KEYWORDS))


def install_listener_hotkey_patch(listener_class) -> None:
    """Use GetAsyncKeyState for control hotkeys, preserving edge detection."""
    if getattr(listener_class, "_async_hotkey_patch_installed", False):
        return

    original_pressed_once = listener_class._pressed_once

    def pressed_once(self, key):
        normalized = "" if key is None else str(key).strip().lower()
        vk = VK_MAP.get(normalized)
        if vk is None:
            return original_pressed_once(self, normalized)

        pressed = bool(user32.GetAsyncKeyState(vk) & 0x8000)
        token = f"win32:{normalized}"
        if pressed:
            if token not in self._previously_pressed:
                self._previously_pressed.add(token)
                return True
        else:
            self._previously_pressed.discard(token)
        return False

    listener_class._pressed_once = pressed_once
    listener_class._async_hotkey_patch_installed = True


def install_patrol_focus_patch(patrol_class) -> None:
    """Allow patrol input only while the foreground window is MapleStory.

    The classic client may recreate its HWND or expose a foreground top-level
    window whose PID differs from the stale handle WGC originally selected. We
    therefore accept, in order: exact HWND, same process, or an explicit
    MapleStory window title. CMD/browser/Desktop remain rejected.
    """
    if getattr(patrol_class, "_process_focus_patch_installed", False):
        return

    def foreground_is_game() -> bool:
        from src.common import config

        capture = getattr(config, "capture", None)
        if capture is None:
            return False

        game_hwnd = int(getattr(capture, "_handle", 0) or 0)
        foreground = int(user32.GetForegroundWindow() or 0)
        if not foreground or not user32.IsWindow(foreground):
            return False

        if game_hwnd and user32.IsWindow(game_hwnd):
            if foreground == game_hwnd:
                return True

            game_pid = _window_process_id(game_hwnd)
            foreground_pid = _window_process_id(foreground)
            if game_pid and foreground_pid and game_pid == foreground_pid:
                return True

        # Conservative fallback for the classic client. The actual foreground
        # window itself must identify as MapleStory; unrelated applications are
        # never accepted merely because the stored capture handle became stale.
        return _title_is_maple(_window_title(foreground))

    patrol_class._foreground_is_game = staticmethod(foreground_is_game)
    patrol_class._process_focus_patch_installed = True
