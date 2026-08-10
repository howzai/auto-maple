"""Robust hotkey polling and game-focus checks for patrol input.

The classic client can consume keyboard events in ways that make the third-party
``keyboard`` package miss Insert while MapleStory is focused.  Windows'
GetAsyncKeyState is used for the core control hotkeys instead.  Patrol focus is
validated by process identity rather than requiring the foreground HWND to be
exactly the same HWND selected by WGC; some game/window configurations expose a
child/owned foreground window belonging to the same MapleStory process.
"""

from __future__ import annotations

import ctypes


user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

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


def _window_process_id(hwnd: int) -> int:
    if not hwnd or not user32.IsWindow(hwnd):
        return 0
    pid = ctypes.c_ulong(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


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
    """Allow input only when foreground belongs to the captured game process."""
    if getattr(patrol_class, "_process_focus_patch_installed", False):
        return

    @staticmethod
    def foreground_is_game() -> bool:
        from src.common import config

        capture = getattr(config, "capture", None)
        if capture is None:
            return False
        game_hwnd = int(getattr(capture, "_handle", 0) or 0)
        foreground = int(user32.GetForegroundWindow() or 0)
        if not game_hwnd or not foreground:
            return False

        # Exact match remains the fastest/safest case.
        if foreground == game_hwnd and bool(user32.IsWindow(game_hwnd)):
            return True

        # Owned/child foreground windows can have a different HWND while still
        # belonging to the same MapleStory process.  Process equality keeps the
        # safety boundary at the game process and will still reject CMD/browser.
        game_pid = _window_process_id(game_hwnd)
        foreground_pid = _window_process_id(foreground)
        return bool(game_pid and foreground_pid and game_pid == foreground_pid)

    patrol_class._foreground_is_game = foreground_is_game
    patrol_class._process_focus_patch_installed = True
