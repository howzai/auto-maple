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
        'Start recording': 'f8',
        'Stop recording': 'f9',
        'Vision debug': 'f10',
        'Emergency stop': 'f12',
    }
    BLOCK_DELAY = 1
    POLL_INTERVAL = 0.02

    def __init__(self):
        super().__init__('controls')
        config.listener = self
        self.enabled = False
        self.ready = False
        self.block_time = 0
        self._previously_pressed = set()
        self.thread = threading.Thread(target=self._main, name='keyboard-listener', daemon=True)

    def start(self):
        print('\n[~] Started keyboard listener')
        self.thread.start()

    def _pressed_once(self, key):
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
                record_start_key = self.config.get('Start recording', 'f8')
                record_stop_key = self.config.get('Stop recording', 'f9')
                debug_key = self.config.get('Vision debug', 'f10')
                if self._pressed_once(emergency_key):
                    self.emergency_stop()
                elif self._pressed_once(record_start_key):
                    self.start_recording()
                elif self._pressed_once(record_stop_key):
                    self.stop_recording()
                elif self._pressed_once(debug_key):
                    self.toggle_vision_debug()
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
    def start_recording():
        recorder = getattr(config, 'data_recorder', None)
        if recorder is None:
            print('\n[!] Data recorder unavailable')
            return
        recorder.start_session()

    @staticmethod
    def stop_recording():
        recorder = getattr(config, 'data_recorder', None)
        if recorder is None:
            print('\n[!] Data recorder unavailable')
            return
        if not recorder.stop_session():
            print('\n[!] Data recorder is not currently running')

    @staticmethod
    def toggle_vision_debug():
        observer = getattr(config, 'scene_observer', None)
        if observer is None:
            print('\n[!] Vision debug unavailable: scene observer is not running')
            return
        observer.toggle_debug()

    @staticmethod
    def emergency_stop():
        config.enabled = False
        if getattr(config, 'bot', None) is not None:
            config.bot.rune_active = False
        recorder = getattr(config, 'data_recorder', None)
        if recorder is not None:
            recorder.stop_session()
        release_all()
        print('\n[!] EMERGENCY STOP: automation disabled, recording stopped, and all keys released')
        try:
            winsound.Beep(392, 180)
            winsound.Beep(262, 300)
        except RuntimeError:
            pass

    @staticmethod
    def _capture_is_usable():
        capture = getattr(config, 'capture', None)
        if capture is None:
            return False, 'capture service is unavailable'
        if not getattr(capture, 'window_found', False):
            return False, 'MapleStory window was not found or is minimized'
        if not getattr(capture, 'calibrated', False):
            return False, 'minimap is not calibrated'
        if not getattr(capture, 'player_found', False):
            return False, 'player marker is not currently visible on the minimap'
        return True, ''

    @staticmethod
    def toggle_enabled():
        config.bot.rune_active = False
        if not config.enabled:
            if not Listener.recalibrate_minimap(timeout=10):
                error = getattr(config.capture, 'last_error', None)
                suffix = f': {error}' if error else ''
                print(f'\n[!] Cannot enable: minimap calibration timed out{suffix}')
                release_all()
                return
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not config.capture.player_found:
                time.sleep(0.02)
            usable, reason = Listener._capture_is_usable()
            if not usable:
                print(f'\n[!] Cannot enable: {reason}')
                release_all()
                return
        config.enabled = not config.enabled
        if not config.enabled:
            release_all()
        utils.print_state()
        try:
            winsound.Beep(784 if config.enabled else 523, 333)
        except RuntimeError:
            pass
        time.sleep(0.267)

    @staticmethod
    def reload_routine():
        if not Listener.recalibrate_minimap(timeout=10):
            print('\n[!] Cannot reload routine: minimap calibration timed out')
            return
        usable, reason = Listener._capture_is_usable()
        if not usable:
            print(f'\n[!] Cannot reload routine: {reason}')
            return
        config.routine.load(config.routine.path)
        try:
            winsound.Beep(523, 200)
            winsound.Beep(659, 200)
            winsound.Beep(784, 200)
        except RuntimeError:
            pass

    @staticmethod
    def recalibrate_minimap(timeout=10):
        capture = getattr(config, 'capture', None)
        if capture is None:
            return False
        capture.calibrated = False
        deadline = time.monotonic() + timeout
        while not capture.calibrated:
            if not capture.thread.is_alive() or time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        if getattr(config, 'gui', None) is not None:
            config.gui.edit.minimap.redraw()
        return True

    @staticmethod
    def record_position():
        capture = getattr(config, 'capture', None)
        if capture is None or not capture.player_found:
            print('\n[!] Cannot record position: player marker is not visible')
            return
        pos = tuple('{:.3f}'.format(round(i, 3)) for i in config.player_pos)
        now = datetime.now().strftime('%I:%M:%S %p')
        config.gui.edit.record.add_entry(now, pos)
        print(f'\n[~] Recorded position ({pos[0]}, {pos[1]}) at {now}')
        time.sleep(0.6)
