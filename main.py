"""Central application startup with bounded initialization and safe shutdown."""

import sys
import time

from src.common import config
from src.common.logging_config import configure_logging, get_logger
from src.common.vkeys import release_all
from src.modules.bot import Bot
from src.modules.capture import Capture
from src.modules.classic_vision_backend import install_classic_vision_backend
from src.modules.classic_player import install_classic_player_fallback
from src.modules.data_recorder import DataRecorder
from src.modules.gui import GUI
from src.modules.input_safety import install_listener_hotkey_patch, install_patrol_focus_patch
from src.modules.listener import Listener
from src.modules.manual_foreground_input import install_manual_foreground_input
from src.modules.notifier import Notifier
from src.modules.patrol_controller import PatrolController
from src.modules.patrol_input_patch import install_patrol_input_patch
from src.modules.scene_observer import SceneObserver
from src.modules.scene_performance import install_scene_performance_patch
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
            raise TimeoutError(f"{display_name} did not become ready within {timeout} seconds")
        time.sleep(0.02)

    logger.info("%s is ready", display_name)


def main():
    """Initialize all modules and start the GUI."""
    configure_logging()
    logger.info("Auto Maple startup requested")

    install_wgc_capture(Capture)
    install_classic_vision_backend(Capture)
    install_classic_player_fallback(Capture)
    install_scene_performance_patch(SceneObserver)
    install_listener_hotkey_patch(Listener)
    install_patrol_focus_patch(PatrolController)
    install_patrol_input_patch(PatrolController)
    install_manual_foreground_input(Listener, PatrolController)

    bot = Bot()
    capture = Capture()
    scene_observer = SceneObserver()
    patrol_controller = PatrolController()
    data_recorder = DataRecorder()
    notifier = Notifier()
    listener = Listener()

    _start_and_wait(bot, "Bot")
    _start_and_wait(capture, "Capture")
    _start_and_wait(scene_observer, "Scene Observer")
    _start_and_wait(patrol_controller, "Patrol Controller")
    _start_and_wait(data_recorder, "Data Recorder")
    _start_and_wait(notifier, "Notifier")
    _start_and_wait(listener, "Listener")

    logger.info("Successfully initialized Auto Maple")
    print("\n[~] Successfully initialized Auto Maple")
    print("[~] Capture backend: Windows Graphics Capture")
    print("[~] Minimap vision: Classic fixed-UI geometry / F11 manual search region")
    print("[~] Main-scene vision: Monster 10Hz + Ladder/Platform 1Hz @ 480px")
    print("[~] Idle mode: main-scene YOLO sleeps until automation or F10 is enabled")
    print("[~] F10 preview: combined Monster / Ladder / Platform cache (5Hz)")
    print("[~] Patrol: Insert starts/stops; Shift attack; Space jump; Down+Space drop; rapid Z loot")
    print("[~] Input: manual-foreground scan-code SendInput; Auto Maple never forces MapleStory to the front")
    print("[~] Safety: patrol input pauses immediately when MapleStory is not the active window")
    print("[~] Press F8 to start dataset recording")
    print("[~] Press F9 to stop dataset recording")
    print("[~] Press F10 to toggle combined vision debug")
    print("[~] Press F11 to draw/save the minimap search region")
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
        patrol = getattr(config, "patrol_controller", None)
        if patrol is not None:
            try:
                patrol.stop()
            except Exception:
                logger.exception("Patrol controller shutdown failed")
        recorder = getattr(config, "data_recorder", None)
        if recorder is not None:
            try:
                recorder.stop()
            except Exception:
                logger.exception("Data recorder shutdown failed")
        observer = getattr(config, "scene_observer", None)
        if observer is not None:
            try:
                observer.stop()
            except Exception:
                logger.exception("Scene observer shutdown failed")
        capture = getattr(config, "capture", None)
        if capture is not None:
            try:
                capture.stop()
            except Exception:
                logger.exception("Capture shutdown failed")
        release_all()
        logger.info("Auto Maple shutdown complete with exit code %s", exit_code)

    sys.exit(exit_code)
