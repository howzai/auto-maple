"""Robust hotkey polling and safe MapleStory focus checks for patrol input."""

from __future__ import annotations

import ctypes
from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)

# IMPORTANT: HWND is pointer-sized on 64-bit Windows. Declare every Win32 API
# used here so ctypes never truncates a handle.
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


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT),
    ]


user32.GetGUIThreadInfo.argtypes = (wintypes.DWORD, ctypes.POINTER(GUITHREADINFO))
user32.GetGUIThreadInfo.restype = wintypes.BOOL

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


def _focus_candidates():
    """Return all Win32 handles that can represent the currently active UI.

    GetForegroundWindow alone is unreliable for some older DirectX clients.
    GetGUIThreadInfo(0) also exposes hwndActive/hwndFocus for the foreground input
    queue, which is often the actual game child window.
    """
    result = []

    def add(hwnd):
        value = int(hwnd or 0)
        if _valid_hwnd(value) and value not in result:
            result.append(value)

    add(user32.GetForegroundWindow())
    info = GUITHREADINFO()
    info.cbSize = ctypes.sizeof(GUITHREADINFO)
    if user32.GetGUIThreadInfo(0, ctypes.byref(info)):
        add(info.hwndActive)
        add(info.hwndFocus)
        add(info.hwndCapture)
        add(info.hwndMenuOwner)
        add(info.hwndMoveSize)

    expanded = []
    for hwnd in result:
        for related in _related_windows(hwnd):
            if related not in expanded:
                expanded.append(related)
    return tuple(expanded)


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
    """Trust the exact WGC target and accept Win32 active/focus child windows."""
    if getattr(patrol_class, "_process_focus_patch_installed", False):
        return

    def foreground_is_game() -> bool:
        from src.common import config

        capture = getattr(config, "capture", None)
        if capture is None:
            return False

        game_hwnd = int(getattr(capture, "_capture_target_hwnd", 0) or 0)
        game_pid = int(getattr(capture, "_capture_target_pid", 0) or 0)
        active_family = _focus_candidates()
        if not active_family:
            return False

        if game_hwnd:
            game_family = _related_windows(game_hwnd)
            if set(active_family).intersection(game_family):
                return True

        if game_pid:
            active_pids = {_window_process_id(hwnd) for hwnd in active_family}
            active_pids.discard(0)
            if game_pid in active_pids:
                return True

        # Conservative fallback: one of the actual active/focus/root-owner HWNDs
        # must explicitly identify as MapleStory. Browser/CMD/Desktop still fail.
        return any(_title_is_maple(_window_title(hwnd)) for hwnd in active_family)

    def focus_debug_text() -> str:
        from src.common import config

        capture = getattr(config, "capture", None)
        game_hwnd = int(getattr(capture, "_capture_target_hwnd", 0) or 0) if capture is not None else 0
        game_pid = int(getattr(capture, "_capture_target_pid", 0) or 0) if capture is not None else 0
        active = _focus_candidates()
        active_text = "; ".join(_describe(hwnd) for hwnd in active) or "none"
        game_text = _describe(game_hwnd) if game_hwnd else "none"
        return (
            f"active=[{active_text}] "
            f"wgc_target=[{game_text}] shared_game_pid={game_pid}"
        )

    patrol_class._foreground_is_game = staticmethod(foreground_is_game)
    patrol_class._focus_debug_text = staticmethod(focus_debug_text)
    patrol_class._process_focus_patch_installed = True
