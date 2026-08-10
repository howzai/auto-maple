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
GA_ROOT = 2
GA_ROOTOWNER = 3
GW_OWNER = 4


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


def _window_class(hwnd: int) -> str:
    if not hwnd or not user32.IsWindow(hwnd):
        return ""
    buffer = ctypes.create_unicode_buffer(256)
    if not user32.GetClassNameW(hwnd, buffer, len(buffer)):
        return ""
    return buffer.value.strip()


def _title_is_maple(title: str) -> bool:
    normalized = (title or "").casefold()
    return bool(normalized and any(keyword.casefold() in normalized for keyword in GAME_TITLE_KEYWORDS))


def _related_windows(hwnd: int):
    if not hwnd or not user32.IsWindow(hwnd):
        return ()
    result = []

    def add(candidate):
        candidate = int(candidate or 0)
        if candidate and user32.IsWindow(candidate) and candidate not in result:
            result.append(candidate)

    add(hwnd)
    add(user32.GetAncestor(hwnd, GA_ROOT))
    add(user32.GetAncestor(hwnd, GA_ROOTOWNER))

    current = hwnd
    for _ in range(8):
        owner = int(user32.GetWindow(current, GW_OWNER) or 0)
        if not owner or owner == current:
            break
        add(owner)
        current = owner

    return tuple(result)


def _describe(hwnd: int) -> str:
    return (
        f"hwnd=0x{int(hwnd or 0):X} pid={_window_process_id(hwnd)} "
        f"class={_window_class(hwnd)!r} title={_window_title(hwnd)!r}"
    )


def install_listener_hotkey_patch(listener_class) -> None:
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
    """Trust the exact HWND/PID selected by MapleCaptureHost, not a guessed Python HWND."""
    if getattr(patrol_class, "_process_focus_patch_installed", False):
        return

    def foreground_is_game() -> bool:
        from src.common import config

        capture = getattr(config, "capture", None)
        if capture is None:
            return False

        game_hwnd = int(getattr(capture, "_capture_target_hwnd", 0) or 0)
        game_pid = int(getattr(capture, "_capture_target_pid", 0) or 0)
        foreground = int(user32.GetForegroundWindow() or 0)
        if not foreground or not user32.IsWindow(foreground):
            return False

        foreground_family = _related_windows(foreground)

        # Primary safety path: use the exact game identity published by the C#
        # process that actually selected the WGC target window.
        if game_hwnd:
            game_family = _related_windows(game_hwnd)
            if set(foreground_family).intersection(game_family):
                return True

        if game_pid:
            foreground_pids = {_window_process_id(hwnd) for hwnd in foreground_family}
            foreground_pids.discard(0)
            if game_pid in foreground_pids:
                return True

        # Conservative fallback only while the shared identity is not available
        # yet. The visible foreground family itself must identify as MapleStory.
        return any(_title_is_maple(_window_title(hwnd)) for hwnd in foreground_family)

    def focus_debug_text() -> str:
        from src.common import config

        capture = getattr(config, "capture", None)
        game_hwnd = int(getattr(capture, "_capture_target_hwnd", 0) or 0) if capture is not None else 0
        game_pid = int(getattr(capture, "_capture_target_pid", 0) or 0) if capture is not None else 0
        foreground = int(user32.GetForegroundWindow() or 0)
        fg_family = _related_windows(foreground)
        fg_text = "; ".join(_describe(hwnd) for hwnd in fg_family) or "none"
        game_text = _describe(game_hwnd) if game_hwnd else "none"
        return (
            f"foreground=[{fg_text}] "
            f"wgc_target=[{game_text}] shared_game_pid={game_pid}"
        )

    patrol_class._foreground_is_game = staticmethod(foreground_is_game)
    patrol_class._focus_debug_text = staticmethod(focus_debug_text)
    patrol_class._process_focus_patch_installed = True
