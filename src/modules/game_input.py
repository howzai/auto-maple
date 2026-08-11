"""Safe keyboard input for the exact MapleStory WGC target.

The classic Unity client ignores WM_KEYDOWN/WM_KEYUP posted with PostMessage.
Instead, focus the exact HWND published by MapleCaptureHost and inject hardware-like
scan-code events with SendInput.  No event is sent unless that exact game window
can be made the active foreground target at send time.
"""

from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes

from src.common import config

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_EXTENDEDKEY = 0x0001
SW_RESTORE = 9

# Set-1 keyboard scan codes.  These are what DirectInput-style games expect.
SCAN = {
    "left": 0x4B,
    "up": 0x48,
    "right": 0x4D,
    "down": 0x50,
    "shift": 0x2A,   # left Shift
    "space": 0x39,
    "z": 0x2C,
}
EXTENDED_KEYS = {"left", "up", "right", "down"}

wintypes.ULONG_PTR = wintypes.WPARAM


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
user32.GetForegroundWindow.argtypes = ()
user32.GetForegroundWindow.restype = wintypes.HWND
user32.IsWindow.argtypes = (wintypes.HWND,)
user32.IsWindow.restype = wintypes.BOOL
user32.IsIconic.argtypes = (wintypes.HWND,)
user32.IsIconic.restype = wintypes.BOOL
user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
user32.ShowWindow.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.BringWindowToTop.argtypes = (wintypes.HWND,)
user32.BringWindowToTop.restype = wintypes.BOOL
user32.SetActiveWindow.argtypes = (wintypes.HWND,)
user32.SetActiveWindow.restype = wintypes.HWND
user32.SetFocus.argtypes = (wintypes.HWND,)
user32.SetFocus.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
user32.AttachThreadInput.restype = wintypes.BOOL
kernel32.GetCurrentThreadId.argtypes = ()
kernel32.GetCurrentThreadId.restype = wintypes.DWORD

_pressed = set()
_lock = threading.RLock()


def _target_hwnd() -> int:
    capture = getattr(config, "capture", None)
    if capture is None:
        return 0
    hwnd = int(getattr(capture, "_capture_target_hwnd", 0) or 0)
    return hwnd if hwnd and user32.IsWindow(wintypes.HWND(hwnd)) else 0


def _thread_id(hwnd: int) -> int:
    if not hwnd:
        return 0
    pid = wintypes.DWORD(0)
    return int(user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid)) or 0)


def _focus_exact_target() -> bool:
    """Make the exact WGC target the input foreground immediately before SendInput."""
    hwnd = _target_hwnd()
    if not hwnd:
        return False
    target = wintypes.HWND(hwnd)
    if int(user32.GetForegroundWindow() or 0) == hwnd:
        return True

    if user32.IsIconic(target):
        user32.ShowWindow(target, SW_RESTORE)

    current_tid = int(kernel32.GetCurrentThreadId() or 0)
    foreground = int(user32.GetForegroundWindow() or 0)
    foreground_tid = _thread_id(foreground)
    target_tid = _thread_id(hwnd)
    attached = []
    try:
        for tid in (foreground_tid, target_tid):
            if tid and current_tid and tid != current_tid and tid not in attached:
                if user32.AttachThreadInput(current_tid, tid, True):
                    attached.append(tid)
        user32.BringWindowToTop(target)
        user32.SetForegroundWindow(target)
        user32.SetActiveWindow(target)
        user32.SetFocus(target)
    finally:
        for tid in reversed(attached):
            user32.AttachThreadInput(current_tid, tid, False)

    # Give the old Unity input queue a tiny amount of time to observe focus.
    deadline = time.monotonic() + 0.08
    while time.monotonic() < deadline:
        if int(user32.GetForegroundWindow() or 0) == hwnd:
            return True
        time.sleep(0.005)
    return False


def _send_scan(key: str, key_up_event: bool) -> bool:
    key = str(key).lower()
    if key not in SCAN:
        return False
    if not _focus_exact_target():
        return False

    flags = KEYEVENTF_SCANCODE
    if key in EXTENDED_KEYS:
        flags |= KEYEVENTF_EXTENDEDKEY
    if key_up_event:
        flags |= KEYEVENTF_KEYUP

    event = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(wVk=0, wScan=SCAN[key], dwFlags=flags, time=0, dwExtraInfo=0),
    )
    return int(user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))) == 1


def key_down(key: str) -> bool:
    key = str(key).lower()
    if not config.enabled:
        return False
    ok = _send_scan(key, False)
    if ok:
        with _lock:
            _pressed.add(key)
    return ok


def key_up(key: str) -> bool:
    key = str(key).lower()
    ok = _send_scan(key, True)
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
        try:
            time.sleep(max(0.0, down_time))
        finally:
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
            _send_scan(key, True)
        except Exception:
            pass
