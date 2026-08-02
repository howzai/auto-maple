"""Low-level keyboard and mouse input helpers.

This module keeps track of keys pressed by Auto Maple so every key can be
released safely when the bot pauses, exits, or encounters an error.
"""

import atexit
import ctypes
import threading
import time
from ctypes import wintypes
from random import random

import win32api
import win32con

from src.common import utils


user32 = ctypes.WinDLL('user32', use_last_error=True)

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

MAPVK_VK_TO_VSC = 0

KEY_MAP = {
    'left': 0x25, 'up': 0x26, 'right': 0x27, 'down': 0x28,
    'backspace': 0x08, 'tab': 0x09, 'enter': 0x0D, 'shift': 0x10,
    'ctrl': 0x11, 'alt': 0x12, 'caps lock': 0x14, 'esc': 0x1B,
    'space': 0x20, 'page up': 0x21, 'page down': 0x22, 'end': 0x23,
    'home': 0x24, 'insert': 0x2D, 'delete': 0x2E,
    '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
    '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
    'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
    'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
    'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
    'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
    'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59,
    'z': 0x5A,
    'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73, 'f5': 0x74,
    'f6': 0x75, 'f7': 0x76, 'f8': 0x77, 'f9': 0x78, 'f10': 0x79,
    'f11': 0x7A, 'f12': 0x7B, 'num lock': 0x90, 'scroll lock': 0x91,
    ';': 0xBA, '=': 0xBB, ',': 0xBC, '-': 0xBD, '.': 0xBE,
    '/': 0xBF, '`': 0xC0, '[': 0xDB, '\\': 0xDC, ']': 0xDD,
    "'": 0xDE,
}

wintypes.ULONG_PTR = wintypes.WPARAM


class KeyboardInput(ctypes.Structure):
    _fields_ = (
        ('wVk', wintypes.WORD),
        ('wScan', wintypes.WORD),
        ('dwFlags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', wintypes.ULONG_PTR),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk, MAPVK_VK_TO_VSC, 0)


class MouseInput(ctypes.Structure):
    _fields_ = (
        ('dx', wintypes.LONG), ('dy', wintypes.LONG),
        ('mouseData', wintypes.DWORD), ('dwFlags', wintypes.DWORD),
        ('time', wintypes.DWORD), ('dwExtraInfo', wintypes.ULONG_PTR),
    )


class HardwareInput(ctypes.Structure):
    _fields_ = (
        ('uMsg', wintypes.DWORD),
        ('wParamL', wintypes.WORD),
        ('wParamH', wintypes.WORD),
    )


class Input(ctypes.Structure):
    class _Input(ctypes.Union):
        _fields_ = (
            ('ki', KeyboardInput),
            ('mi', MouseInput),
            ('hi', HardwareInput),
        )

    _anonymous_ = ('_input',)
    _fields_ = (('type', wintypes.DWORD), ('_input', _Input))


LPINPUT = ctypes.POINTER(Input)


def err_check(result, _, args):
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return args


user32.SendInput.errcheck = err_check
user32.SendInput.argtypes = (wintypes.UINT, LPINPUT, ctypes.c_int)

_pressed_keys = set()
_pressed_keys_lock = threading.RLock()


def _validate_key(key):
    normalized = str(key).lower()
    if normalized not in KEY_MAP:
        raise ValueError(f"Invalid keyboard input: '{key}'.")
    return normalized


def _send_key(key, key_up_event=False):
    flags = KEYEVENTF_KEYUP if key_up_event else 0
    event = Input(
        type=INPUT_KEYBOARD,
        ki=KeyboardInput(wVk=KEY_MAP[key], dwFlags=flags),
    )
    user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(event))


@utils.run_if_enabled
def key_down(key):
    """Press KEY down and remember it for emergency release."""
    key = _validate_key(key)
    with _pressed_keys_lock:
        _send_key(key)
        _pressed_keys.add(key)


def key_up(key):
    """Release KEY even when the bot has already been disabled."""
    key = _validate_key(key)
    with _pressed_keys_lock:
        _send_key(key, key_up_event=True)
        _pressed_keys.discard(key)


def release_all():
    """Release every key currently held by Auto Maple.

    This function intentionally bypasses ``run_if_enabled`` so it remains usable
    during emergency shutdown and exception handling.
    """
    with _pressed_keys_lock:
        keys = tuple(_pressed_keys)
        _pressed_keys.clear()

    for key in keys:
        try:
            _send_key(key, key_up_event=True)
        except OSError as exc:
            print(f"\n[!] Failed to release key '{key}': {exc}")


@utils.run_if_enabled
def press(key, n, down_time=0.05, up_time=0.1):
    """Press KEY N times with small randomized timing."""
    if n < 1:
        return

    for _ in range(n):
        key_down(key)
        try:
            time.sleep(max(0, down_time) * (0.8 + 0.4 * random()))
        finally:
            key_up(key)
        time.sleep(max(0, up_time) * (0.8 + 0.4 * random()))


@utils.run_if_enabled
def click(position, button='left'):
    """Simulate a mouse click at POSITION."""
    if button not in ('left', 'right'):
        raise ValueError(f"'{button}' is not a valid mouse button.")

    if button == 'left':
        down_event = win32con.MOUSEEVENTF_LEFTDOWN
        up_event = win32con.MOUSEEVENTF_LEFTUP
    else:
        down_event = win32con.MOUSEEVENTF_RIGHTDOWN
        up_event = win32con.MOUSEEVENTF_RIGHTUP

    x, y = int(position[0]), int(position[1])
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(down_event, x, y, 0, 0)
    win32api.mouse_event(up_event, x, y, 0, 0)


atexit.register(release_all)
