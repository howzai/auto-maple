"""Robust hotkey polling and safe MapleStory focus checks for patrol input."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# IMPORTANT: HWND is pointer-sized on 64-bit Windows. Declare every Win32 API
# used here so ctypes never truncates a handle.
user32.GetForegroundWindow.argtypes = ()
user32.GetForegroundWindow.restype = wintypes.HWND
user32.IsWindow.argtypes = (wintypes.HWND,)
user32.IsWindow.restype = wintypes.BOOL
user32.IsIconic.argtypes = (wintypes.HWND,)
user32.IsIconic.restype = wintypes.BOOL
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
user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.BringWindowToTop.argtypes = (wintypes.HWND,)
user32.BringWindowToTop.restype = wintypes.BOOL
user32.SetActiveWindow.argtypes = (wintypes.HWND,)
user32.SetActiveWindow.restype = wintypes.HWND
user32.SetFocus.argtypes = (wintypes.HWND,)
user32.SetFocus.restype = wintypes.HWND
user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
user32.ShowWindow.restype = wintypes.BOOL
user32.AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
user32.AttachThreadInput.restype = wintypes.BOOL
kernel32.GetCurrentThreadId.argtypes = ()
kernel32.GetCurrentThreadId.restype = wintypes.DWORD


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
SW_RESTORE = 9


def _valid_hwnd(hwnd: int) -> bool:
    return bool(hwnd and user32.IsWindow(wintypes.HWND(int(hwnd))))


def _window_thread_process(hwnd: int) -> tuple[int, int]:
    if not _valid_hwnd(hwnd):
        return 0, 0
    pid = wintypes.DWORD(0)
    thread_id = user32.GetWindowThreadProcessId(wintypes.HWND(int(hwnd)), ctypes.byref(pid))
    return int(thread_id or 0), int(pid.value)


def _window_process_id(hwnd: int) -> int:
    return _window_thread_process(hwnd)[1]


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


def _capture_game_identity():
    from src.common import config

    capture = getattr(config, "capture", None)
    if capture is None:
        return 0, 0
    return (
        int(getattr(capture, "_capture_target_hwnd", 0) or 0),
        int(getattr(capture, "_capture_target_pid", 0) or 0),
    )


def _active_matches_game() -> bool:
    game_hwnd, game_pid = _capture_game_identity()
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

    return any(_title_is_maple(_window_title(hwnd)) for hwnd in active_family)


def _activate_capture_target() -> bool:
    """Bring only the exact WGC MapleStory target to the foreground.

    This is intentionally attempted only when the user enables automation with
    Insert. Patrol never steals focus back after the user switches to another
    application; it simply pauses and releases all keys.
    """
    game_hwnd, _game_pid = _capture_game_identity()
    if not _valid_hwnd(game_hwnd):
        return False
    if _active_matches_game():
        return True

    target = wintypes.HWND(game_hwnd)
    if user32.IsIconic(target):
        user32.ShowWindow(target, SW_RESTORE)

    foreground = int(user32.GetForegroundWindow() or 0)
    current_tid = int(kernel32.GetCurrentThreadId() or 0)
    foreground_tid, _ = _window_thread_process(foreground)
    game_tid, _ = _window_thread_process(game_hwnd)

    attached = []
    try:
        for other_tid in (foreground_tid, game_tid):
            if other_tid and current_tid and other_tid != current_tid and other_tid not in attached:
                if user32.AttachThreadInput(current_tid, other_tid, True):
                    attached.append(other_tid)

        user32.BringWindowToTop(target)
        user32.SetForegroundWindow(target)
        user32.SetActiveWindow(target)
        user32.SetFocus(target)
    finally:
        for other_tid in reversed(attached):
            user32.AttachThreadInput(current_tid, other_tid, False)

    deadline = time.monotonic() + 0.45
    while time.monotonic() < deadline:
        if _active_matches_game():
            return True
        time.sleep(0.02)
    return False


def install_listener_hotkey_patch(listener_class) -> None:
    if getattr(listener_class, "_async_hotkey_patch_installed", False):
        return

    original_pressed_once = listener_class._pressed_once
    original_toggle_enabled = listener_class.toggle_enabled

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

    def toggle_enabled():
        from src.common import config
        from src.common.vkeys import release_all

        # When starting, the hotkey may have been pressed while Windows Terminal
        # still owns foreground focus. Explicitly focus the exact WGC target once.
        if not config.enabled:
            if not _activate_capture_target():
                release_all()
                print("\n[!] Cannot enable patrol: unable to focus the WGC MapleStory target")
                game_hwnd, game_pid = _capture_game_identity()
                print(
                    f"[FOCUS] target={_describe(game_hwnd) if game_hwnd else 'none'} "
                    f"shared_game_pid={game_pid} active="
                    + ("; ".join(_describe(hwnd) for hwnd in _focus_candidates()) or "none")
                )
                return

        original_toggle_enabled()

        # Reassert the target once after minimap recalibration/startup work. This
        # does not run continuously, so switching away later still pauses safely.
        if config.enabled:
            _activate_capture_target()

    listener_class._pressed_once = pressed_once
    listener_class.toggle_enabled = staticmethod(toggle_enabled)
    listener_class._async_hotkey_patch_installed = True


def install_patrol_focus_patch(patrol_class) -> None:
    """Trust the exact WGC target and accept Win32 active/focus child windows."""
    if getattr(patrol_class, "_process_focus_patch_installed", False):
        return

    def foreground_is_game() -> bool:
        return _active_matches_game()

    def focus_debug_text() -> str:
        game_hwnd, game_pid = _capture_game_identity()
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
