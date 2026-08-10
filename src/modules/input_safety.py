"""Robust hotkey polling and safe MapleStory focus checks for patrol input."""

from __future__ import annotations

import ctypes
from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)

# IMPORTANT: on 64-bit Windows an HWND is pointer-sized.  ctypes defaults the
# return type of an undeclared Win32 function to c_int, which truncates HWNDs and
# makes IsWindow/GetWindowText/GetWindowThreadProcessId fail even though the game
# is visibly in the foreground.  Declare every window API used here explicitly.
user32.GetForegroundWindow.argtypes = ()
user32.GetForegroundWindow.restype = wintypes.HWND
user32.IsWindow.argtypes = (wintypes.HWND,)
user32.IsWindow.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.GetClassNameW.restype = ctypes.c_int
user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
user32.GetAncestor.restype = wintypes.HWND
user32.GetWindow.argtypes = (wintypes.HWND, wintypes.UINT)
user32.GetWindow.restype = wintypes.HWND
user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
user32.GetAsyncKeyState.restype = ctypes.c_short

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


def _valid_hwnd(hwnd: int) -> bool:
    return bool(hwnd and user32.IsWindow(wintypes.HWND(int(hwnd))))


def _window_process_id(hwnd: int) -> int:
    if not _valid_hwnd(hwnd):
        return 0
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(wintypes.HWND(int(hwnd)), ctypes.byref(pid))
    return int(pid.value)


def _window_title(hwnd: int) -> str:
    if not _valid_hwnd(hwnd):
        return ""
    handle = wintypes.HWND(int(hwnd))
    length = int(user32.GetWindowTextLengthW(handle) or 0)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(handle, buffer, len(buffer))
    return buffer.value.strip()


def _window_class(hwnd: int) -> str:
    if not _valid_hwnd(hwnd):
        return ""
    buffer = ctypes.create_unicode_buffer(256)
    if not user32.GetClassNameW(wintypes.HWND(int(hwnd)), buffer, len(buffer)):
        return ""
    return buffer.value.strip()


def _title_is_maple(title: str) -> bool:
    normalized = (title or "").casefold()
    return bool(normalized and any(keyword.casefold() in normalized for keyword in GAME_TITLE_KEYWORDS))


def _related_windows(hwnd: int):
    if not _valid_hwnd(hwnd):
        return ()
    result = []

    def add(candidate):
        candidate = int(candidate or 0)
        if _valid_hwnd(candidate) and candidate not in result:
            result.append(candidate)

    handle = wintypes.HWND(int(hwnd))
    add(hwnd)
    add(user32.GetAncestor(handle, GA_ROOT))
    add(user32.GetAncestor(handle, GA_ROOTOWNER))

    current = int(hwnd)
    for _ in range(8):
        owner = int(user32.GetWindow(wintypes.HWND(current), GW_OWNER) or 0)
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
    """Trust the exact HWND/PID selected by MapleCaptureHost."""
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
        if not _valid_hwnd(foreground):
            return False

        foreground_family = _related_windows(foreground)

        if game_hwnd:
            game_family = _related_windows(game_hwnd)
            if set(foreground_family).intersection(game_family):
                return True

        if game_pid:
            foreground_pids = {_window_process_id(hwnd) for hwnd in foreground_family}
            foreground_pids.discard(0)
            if game_pid in foreground_pids:
                return True

        # Safe fallback: the actual foreground/root-owner family itself must
        # explicitly identify as MapleStory.  Browser/CMD/Desktop still fail.
        return any(_title_is_maple(_window_title(hwnd)) for hwnd in foreground_family)

    def focus_debug_text() -> str:
        from src.common import config

        capture = getattr(config, "capture", None)
        game_hwnd = int(getattr(capture, "_capture_target_hwnd", 0) or 0) if capture is not None else 0
        game_pid = int(getattr(capture, "_capture_target_pid", 0) or 0) if capture is not None else 0
        foreground = int(user32.GetForegroundWindow() or 0)
        fg_family = _related_windows(foreground)
        fg_text = "; ".join(_describe(hwnd) for hwnd in fg_family) or _describe(foreground)
        game_text = _describe(game_hwnd) if game_hwnd else "none"
        return (
            f"foreground=[{fg_text}] "
            f"wgc_target=[{game_text}] shared_game_pid={game_pid}"
        )

    patrol_class._foreground_is_game = staticmethod(foreground_is_game)
    patrol_class._focus_debug_text = staticmethod(focus_debug_text)
    patrol_class._process_focus_patch_installed = True
