"""Central application startup with bounded initialization and safe shutdown."""

import sys
import time

from src.common import config
from src.common.logging_config import configure_logging, get_logger
from src.common.vkeys import release_all
from src.modules.bot import Bot
from src.modules.capture import Capture
from src.modules.classic_minimap import install_classic_minimap_fallback
from src.modules.classic_player import install_classic_player_fallback
from src.modules.gui import GUI
from src.modules.listener import Listener
from src.modules.notifier import Notifier
from src.modules.wgc_capture_backend import install_wgc_capture


STARTUP_TIMEOUT_SECONDS = 30
logger = get_logger("main")


def _start_and_wait(component, display_name, timeout=STARTUP_TIMEOUT_SECONDS):
    """Start COMPONENT and wait for readiness without hanging forever."""
    logger.info("Starting %s", display_name)
    component.start()
    deadline = time.monotonic() + timeout

    while not component.ready:
        thread = getattr(component, "thread", None)
        if thread is not None and not thread.is_alive():
            raise RuntimeError(f"{display_name} stopped during initialization")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{display_name} did not become ready within {timeout} seconds"
            )
        time.sleep(0.02)

    logger.info("%s is ready", display_name)


def main():
    """Initialize all modules and start the GUI."""
    configure_logging()
    logger.info("Auto Maple startup requested")

    # Windows Graphics Capture is the only frame source on this branch. It is
    # installed before regional minimap/player fallbacks and before Capture is
    # instantiated. There is intentionally no desktop-capture fallback.
    install_wgc_capture(Capture)
    install_classic_minimap_fallback(Capture)
    install_classic_player_fallback(Capture)

    bot = Bot()
    capture = Capture()
    notifier = Notifier()
    listener = Listener()

    _start_and_wait(bot, "Bot")
    _start_and_wait(capture, "Capture")
    _start_and_wait(notifier, "Notifier")
    _start_and_wait(listener, "Listener")

    logger.info("Successfully initialized Auto Maple")
    print("\n[~] Successfully initialized Auto Maple")
    print("[~] Capture backend: Windows Graphics Capture")
    print("[~] Press F12 at any time for emergency stop")

    gui = GUI()
    gui.start()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Auto Maple interrupted by user")
        print("\n[~] Auto Maple interrupted by user")
    except Exception:
        config.enabled = False
        exit_code = 1
        logger.exception("Auto Maple failed to start")
        print("\n[!] Auto Maple failed to start; see logs/auto-maple.log")
    finally:
        config.enabled = False
        capture = getattr(config, "capture", None)
        if capture is not None:
            try:
                capture.stop()
            except Exception:
                logger.exception("Capture shutdown failed")
        release_all()
        logger.info("Auto Maple shutdown complete with exit code %s", exit_code)

    sys.exit(exit_code)
