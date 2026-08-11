"""Window-targeted keyboard input for MapleStory patrol.

Patrol keys are posted only to the exact HWND published by MapleCaptureHost.  This
avoids sending synthetic input to Chrome/Terminal/Desktop and does not modify the
system-wide physical key state.  It also works when Windows refuses to make the
classic client the official foreground window.
"""

from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes

from src.common import config


user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.IsWindow.argtypes = (wintypes.HWND,)
user32.IsWindow.restype = wintypes.BOOL
user32.PostMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
user32.PostMessageW.restype = wintypes.BOOL
user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
user32.MapVirtualKeyW.restype = wintypes.UINT

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
MAPVK_VK_TO_VSC = 0

VK = {
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "shift": 0x10,
    "space": 0x20,
    "z": 0x5A,
}

EXTENDED_KEYS = {"left", "up", "right", "down"}

_pressed = set()
_lock = threading.RLock()


def _target_hwnd() -> int:
    capture = getattr(config, "capture", None)
    if capture is None:
        return 0
    hwnd = int(getattr(capture, "_capture_target_hwnd", 0) or 0)
    if not hwnd or not user32.IsWindow(wintypes.HWND(hwnd)):
        return 0
    return hwnd


def _lparam(key: str, key_up: bool) -> int:
    vk = VK[key]
    scan = int(user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) or 0)
    value = 1 | ((scan & 0xFF) << 16)
    if key in EXTENDED_KEYS:
        value |= 1 << 24
    if key_up:
        value |= (1 << 30) | (1 << 31)
    return value


def key_down(key: str) -> bool:
    key = str(key).lower()
    if key not in VK or not config.enabled:
        return False
    hwnd = _target_hwnd()
    if not hwnd:
        return False
    ok = bool(user32.PostMessageW(wintypes.HWND(hwnd), WM_KEYDOWN, VK[key], _lparam(key, False)))
    if ok:
        with _lock:
            _pressed.add(key)
    return ok


def key_up(key: str) -> bool:
    key = str(key).lower()
    if key not in VK:
        return False
    hwnd = _target_hwnd()
    if not hwnd:
        with _lock:
            _pressed.discard(key)
        return False
    ok = bool(user32.PostMessageW(wintypes.HWND(hwnd), WM_KEYUP, VK[key], _lparam(key, True)))
    with _lock:
        _pressed.discard(key)
    return ok


def press(key: str, n: int = 1, down_time: float = 0.05, up_time: float = 0.03) -> bool:
    if n < 1:
        return False
    sent = False
    for _ in range(n):
        if not key_down(key):
            return sent
        sent = True
        time.sleep(max(0.0, down_time))
        key_up(key)
        time.sleep(max(0.0, up_time))
    return sent


def combo(first: str, second: str, first_lead: float = 0.03, hold: float = 0.08) -> bool:
    if not key_down(first):
        return False
    try:
        time.sleep(max(0.0, first_lead))
        if not key_down(second):
            return False
        try:
            time.sleep(max(0.0, hold))
        finally:
            key_up(second)
    finally:
        key_up(first)
    return True


def release_all() -> None:
    with _lock:
        keys = tuple(_pressed)
        _pressed.clear()
    for key in keys:
        try:
            key_up(key)
        except Exception:
            pass
