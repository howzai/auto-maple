"""Robust hotkey polling and safe MapleStory focus checks for patrol input."""

from __future__ import annotations

import ctypes
from ctypes import wintypes


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
    """Return HWND plus its root/root-owner/owner chain, deduplicated."""
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
        f"hwnd={int(hwnd or 0)} pid={_window_process_id(hwnd)} "
        f"class={_window_class(hwnd)!r} title={_window_title(hwnd)!r}"
    )


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
    """Allow patrol input only while foreground belongs to the MapleStory window family."""
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

        foreground_family = _related_windows(foreground)
        game_family = _related_windows(game_hwnd)

        # Direct/root/owner relationship catches games that expose a child or
        # owned top-level window as the foreground HWND.
        if set(foreground_family).intersection(game_family):
            return True

        game_pids = {_window_process_id(hwnd) for hwnd in game_family}
        game_pids.discard(0)
        foreground_pids = {_window_process_id(hwnd) for hwnd in foreground_family}
        foreground_pids.discard(0)
        if game_pids.intersection(foreground_pids):
            return True

        # Check every related foreground title, not just GetForegroundWindow().
        # Some classic DirectX clients expose an untitled child while the root
        # owner carries the visible MapleStory title.
        if any(_title_is_maple(_window_title(hwnd)) for hwnd in foreground_family):
            return True

        return False

    def focus_debug_text() -> str:
        from src.common import config

        capture = getattr(config, "capture", None)
        game_hwnd = int(getattr(capture, "_handle", 0) or 0) if capture is not None else 0
        foreground = int(user32.GetForegroundWindow() or 0)
        fg_family = _related_windows(foreground)
        game_family = _related_windows(game_hwnd)
        fg_text = "; ".join(_describe(hwnd) for hwnd in fg_family) or "none"
        game_text = "; ".join(_describe(hwnd) for hwnd in game_family) or "none"
        return f"foreground=[{fg_text}] captured=[{game_text}]"

    patrol_class._foreground_is_game = staticmethod(foreground_is_game)
    patrol_class._focus_debug_text = staticmethod(focus_debug_text)
    patrol_class._process_focus_patch_installed = True
