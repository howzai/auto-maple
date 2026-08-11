"""Manual-foreground input mode for the classic MapleStory client.

Windows/Unity can react badly when automation repeatedly calls SetForegroundWindow,
SetFocus, or BringWindowToTop.  In this mode Auto Maple never steals focus.  The
user brings MapleStory to the foreground manually, and patrol input is injected
only while Windows reports the WGC game window/process as active.
"""

from __future__ import annotations

import time
import winsound

from src.common import config, utils
from src.common.vkeys import release_all
from src.modules import game_input, input_safety


def install_manual_foreground_input(listener_class, patrol_class) -> None:
    if getattr(listener_class, "_manual_foreground_input_installed", False):
        return

    def game_is_active() -> bool:
        return bool(input_safety._active_matches_game())

    # Never call SetForegroundWindow/SetFocus before each injected key.  SendInput
    # is allowed only when MapleStory is already the active window/process.
    game_input._focus_exact_target = game_is_active
    patrol_class._foreground_is_game = staticmethod(game_is_active)

    def toggle_enabled():
        # Stop is always allowed from the global Insert hotkey.
        if config.enabled:
            config.enabled = False
            release_all()
            try:
                game_input.release_all()
            except Exception:
                pass
            utils.print_state()
            try:
                winsound.Beep(523, 333)
            except RuntimeError:
                pass
            time.sleep(0.267)
            return

        # Starting must never steal focus.  Ask the user to click MapleStory first.
        if not game_is_active():
            release_all()
            print("\n[!] Cannot enable patrol: MapleStory is not the active window")
            print("[~] Click inside MapleStory, then press Insert again. Auto Maple will not force the game to the front.")
            return

        config.bot.rune_active = False
        if not listener_class.recalibrate_minimap(timeout=10):
            error = getattr(config.capture, "last_error", None)
            suffix = f": {error}" if error else ""
            print(f"\n[!] Cannot enable: minimap calibration timed out{suffix}")
            release_all()
            return

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not config.capture.player_found:
            time.sleep(0.02)

        usable, reason = listener_class._capture_is_usable()
        if not usable:
            print(f"\n[!] Cannot enable: {reason}")
            release_all()
            return

        # Re-check after calibration because GUI/capture work can change focus.
        if not game_is_active():
            release_all()
            print("\n[!] Patrol remains disabled: MapleStory lost focus during startup")
            print("[~] Click inside MapleStory and press Insert again.")
            return

        config.enabled = True
        utils.print_state()
        print("[~] Input mode: manual foreground + scan-code SendInput (no focus stealing)")
        try:
            winsound.Beep(784, 333)
        except RuntimeError:
            pass
        time.sleep(0.267)

    listener_class.toggle_enabled = staticmethod(toggle_enabled)
    listener_class._manual_foreground_input_installed = True
