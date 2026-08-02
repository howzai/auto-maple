"""Central application startup with bounded initialization and safe shutdown."""

import sys
import time
import traceback

from src.common import config
from src.common.vkeys import release_all
from src.modules.bot import Bot
from src.modules.capture import Capture
from src.modules.gui import GUI
from src.modules.listener import Listener
from src.modules.notifier import Notifier


STARTUP_TIMEOUT_SECONDS = 30


def _start_and_wait(component, display_name, timeout=STARTUP_TIMEOUT_SECONDS):
    """Start COMPONENT and wait for readiness without hanging forever."""
    component.start()
    deadline = time.monotonic() + timeout

    while not component.ready:
        thread = getattr(component, 'thread', None)
        if thread is not None and not thread.is_alive():
            raise RuntimeError(f'{display_name} stopped during initialization')
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f'{display_name} did not become ready within {timeout} seconds'
            )
        time.sleep(0.02)


def main():
    """Initialize all modules and start the GUI."""
    bot = Bot()
    capture = Capture()
    notifier = Notifier()
    listener = Listener()

    _start_and_wait(bot, 'Bot')
    _start_and_wait(capture, 'Capture')
    _start_and_wait(notifier, 'Notifier')
    _start_and_wait(listener, 'Listener')

    print('\n[~] Successfully initialized Auto Maple')
    print('[~] Press F12 at any time for emergency stop')

    gui = GUI()
    gui.start()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[~] Auto Maple interrupted by user')
    except Exception as exc:
        config.enabled = False
        print(f'\n[!] Auto Maple failed to start: {exc}')
        traceback.print_exc()
        sys.exit_code = 1
    finally:
        config.enabled = False
        release_all()
