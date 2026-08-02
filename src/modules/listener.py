"""A keyboard listener to track user inputs and emergency-stop requests."""

import threading
import time
import winsound
from datetime import datetime

import keyboard as kb

from src.common import config, utils
from src.common.interfaces import Configurable
from src.common.vkeys import release_all


class Listener(Configurable):
    DEFAULT_CONFIG = {
        'Start/stop': 'insert',
        'Reload routine': 'f6',
        'Record position': 'f7',
        'Emergency stop': 'f12',
    }
    BLOCK_DELAY = 1
    POLL_INTERVAL = 0.02

    def __init__(self):
        """Initialize the listener thread and key edge-detection state."""
        super().__init__('controls')
        config.listener = self

        self.enabled = False
        self.ready = False
        self.block_time = 0
        self._previously_pressed = set()
        self.thread = threading.Thread(
            target=self._main,
            name='keyboard-listener',
            daemon=True,
        )

    def start(self):
        print('\n[~] Started keyboard listener')
        self.thread.start()

    def _pressed_once(self, key):
        """Return True only on the transition from released to pressed."""
        key = str(key).lower()
        pressed = kb.is_pressed(key)
        if pressed:
            if key not in self._previously_pressed:
                self._previously_pressed.add(key)
                return True
        else:
            self._previously_pressed.discard(key)
        return False

    def _main(self):
        self.ready = True
        while True:
            try:
                emergency_key = self.config.get('Emergency stop', 'f12')
                if self._pressed_once(emergency_key):
                    self.emergency_stop()
                elif self.enabled:
                    if self._pressed_once(self.config['Start/stop']):
                        Listener.toggle_enabled()
                    elif self._pressed_once(self.config['Reload routine']):
                        Listener.reload_routine()
                    elif self.restricted_pressed('Record position'):
                        Listener.record_position()
            except Exception as exc:
                config.enabled = False
                release_all()
                print(f'\n[!] Keyboard listener error: {exc}')
                time.sleep(0.5)
            time.sleep(self.POLL_INTERVAL)

    def restricted_pressed(self, action):
        key = self.config[action]
        if self._pressed_once(key):
            if not config.enabled:
                return True
            now = time.time()
            if now - self.block_time > Listener.BLOCK_DELAY:
                print(f"\n[!] Cannot use '{action}' while Auto Maple is enabled")
                self.block_time = now
        return False

    @staticmethod
    def emergency_stop():
        """Immediately disable automation and release every held key."""
        config.enabled = False
        if getattr(config, 'bot', None) is not None:
            config.bot.rune_active = False
        release_all()
        print('\n[!] EMERGENCY STOP: automation disabled and all keys released')
        try:
            winsound.Beep(392, 180)
            winsound.Beep(262, 300)
        except RuntimeError:
            pass

    @staticmethod
    def toggle_enabled():
        config.bot.rune_active = False

        if not config.enabled:
            if not Listener.recalibrate_minimap(timeout=10):
                print('\n[!] Cannot enable: minimap calibration timed out')
                release_all()
                return

        config.enabled = not config.enabled
        if not config.enabled:
            release_all()
        utils.print_state()

        if config.enabled:
            winsound.Beep(784, 333)
        else:
            winsound.Beep(523, 333)
        time.sleep(0.267)

    @staticmethod
    def reload_routine():
        if not Listener.recalibrate_minimap(timeout=10):
            print('\n[!] Cannot reload routine: minimap calibration timed out')
            return

        config.routine.load(config.routine.path)
        winsound.Beep(523, 200)
        winsound.Beep(659, 200)
        winsound.Beep(784, 200)

    @staticmethod
    def recalibrate_minimap(timeout=10):
        config.capture.calibrated = False
        deadline = time.monotonic() + timeout
        while not config.capture.calibrated:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)

        if getattr(config, 'gui', None) is not None:
            config.gui.edit.minimap.redraw()
        return True

    @staticmethod
    def record_position():
        pos = tuple('{:.3f}'.format(round(i, 3)) for i in config.player_pos)
        now = datetime.now().strftime('%I:%M:%S %p')
        config.gui.edit.record.add_entry(now, pos)
        print(f'\n[~] Recorded position ({pos[0]}, {pos[1]}) at {now}')
        time.sleep(0.6)
