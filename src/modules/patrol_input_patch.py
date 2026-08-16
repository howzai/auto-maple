"""Route patrol key actions through the USB HID keyboard bridge."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from src.common import config
from src.modules import game_input


user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.IsWindow.argtypes = (wintypes.HWND,)
user32.IsWindow.restype = wintypes.BOOL


def _game_target_ready() -> bool:
    capture = getattr(config, "capture", None)
    if capture is None:
        return False
    hwnd = int(getattr(capture, "_capture_target_hwnd", 0) or 0)
    return bool(hwnd and user32.IsWindow(wintypes.HWND(hwnd)))


def install_patrol_input_patch(patrol_class) -> None:
    if getattr(patrol_class, "_targeted_input_patch_installed", False):
        return

    def safe_press(self, key: str, down_time: float = 0.04, up_time: float = 0.02) -> bool:
        if not config.enabled or not _game_target_ready() or not game_input.is_ready():
            return False
        return game_input.press(key, 1, down_time=down_time, up_time=up_time)

    def safe_key_down(self, key: str) -> bool:
        if not config.enabled or not _game_target_ready() or not game_input.is_ready():
            return False
        return game_input.key_down(key)

    def safe_key_up(self, key: str) -> bool:
        if not game_input.is_ready():
            return False
        return game_input.key_up(key)

    def safe_combo(self, first: str, second: str, first_lead: float = 0.025, hold: float = 0.08) -> bool:
        if not config.enabled or not _game_target_ready() or not game_input.is_ready():
            return False
        return game_input.combo(first, second, first_lead=first_lead, hold=hold)

    patrol_class._safe_press = safe_press
    patrol_class._safe_key_down = safe_key_down
    patrol_class._safe_key_up = safe_key_up
    patrol_class._safe_combo = safe_combo

    patrol_class.ATTACK_INTERVAL = 0.24
    patrol_class.ATTACK_KEY_DOWN_TIME = 0.12

    original_combat = patrol_class._combat

    def combat_with_stable_attack(self, snapshot, anchor, now: float) -> bool:
        original_safe = self._safe_press

        def timed_safe(key: str, down_time: float = 0.04, up_time: float = 0.02):
            if str(key).lower() == self.ATTACK_KEY:
                return game_input.press(
                    self.ATTACK_KEY,
                    1,
                    down_time=self.ATTACK_KEY_DOWN_TIME,
                    up_time=0.04,
                )
            return original_safe(key, down_time=down_time, up_time=up_time)

        self._safe_press = timed_safe
        try:
            return original_combat(self, snapshot, anchor, now)
        finally:
            self._safe_press = original_safe

    patrol_class._combat = combat_with_stable_attack

    original_stop = patrol_class.stop

    def stop(self):
        try:
            game_input.release_all()
        finally:
            original_stop(self)

    patrol_class.stop = stop
    patrol_class._targeted_input_patch_installed = True
