"""Install the Windows Graphics Capture backend on the existing Capture class.

The backend launches MapleCaptureHost on demand and consumes only frames published
through the named shared-memory mapping. It deliberately has no MSS/PrintWindow
fallback: when WGC is unavailable, automation pauses instead of interpreting pixels
from Chrome, CMD, or another overlapping desktop window as MapleStory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

from src.modules.windows_graphics_capture import WindowsGraphicsCaptureReader


HOST_START_TIMEOUT = 8.0
HOST_RESTART_DELAY = 2.0


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _host_command() -> Optional[list[str]]:
    root = _project_root()
    release_root = root / "capture_host" / "bin" / "Release"

    candidates = [
        release_root
        / "net8.0-windows10.0.19041.0"
        / "win-x64"
        / "MapleCaptureHost.exe",
        root / "capture_host" / "publish" / "MapleCaptureHost.exe",
        root / "MapleCaptureHost.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return [str(candidate)]

    dll_candidates = [
        release_root
        / "net8.0-windows10.0.19041.0"
        / "win-x64"
        / "MapleCaptureHost.dll",
        root / "capture_host" / "publish" / "MapleCaptureHost.dll",
    ]
    for candidate in dll_candidates:
        if candidate.is_file():
            return ["dotnet", str(candidate)]
    return None


class CaptureHostController:
    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen] = None
        self.last_start_attempt = 0.0
        self.last_error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def ensure_running(self) -> bool:
        if self.running:
            return True

        now = time.monotonic()
        if now - self.last_start_attempt < HOST_RESTART_DELAY:
            return False
        self.last_start_attempt = now

        command = _host_command()
        if command is None:
            self.last_error = (
                "MapleCaptureHost was not found. Build capture_host with "
                "'dotnet build -c Release'."
            )
            return False

        creationflags = 0
        if os.name == "nt":
            creationflags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )

        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(_project_root()),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self.last_error = None
            return True
        except (OSError, subprocess.SubprocessError) as exc:
            self.process = None
            self.last_error = f"Unable to start MapleCaptureHost: {exc}"
            return False

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def install_wgc_capture(capture_class) -> None:
    """Patch Capture to source every screenshot from Windows Graphics Capture."""
    if getattr(capture_class, "_wgc_capture_patch_installed", False):
        return

    original_init = capture_class.__init__
    original_stop = capture_class.stop
    original_capture_and_track = capture_class._capture_and_track

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.fps = 0.0
        self.capture_backend = "Windows Graphics Capture: waiting"
        self._wgc_reader = WindowsGraphicsCaptureReader()
        self._wgc_host = CaptureHostController()
        self._wgc_frame_times = deque(maxlen=180)
        self._wgc_wait_started = 0.0

    def patched_stop(self):
        try:
            original_stop(self)
        finally:
            reader = getattr(self, "_wgc_reader", None)
            if reader is not None:
                reader.close()
            host = getattr(self, "_wgc_host", None)
            if host is not None:
                host.stop()

    def patched_screenshot(self, delay: float = 1.0):
        host = self._wgc_host
        reader = self._wgc_reader

        if not host.ensure_running():
            self.capture_backend = "Windows Graphics Capture: host unavailable"
            self.last_error = host.last_error
            if delay:
                time.sleep(min(delay, 0.25))
            return None

        frame = reader.read_latest()
        if frame is None:
            self.capture_backend = "Windows Graphics Capture: waiting for frame"
            self.last_error = reader.last_error or "Waiting for Windows Graphics Capture frame"
            if not self._wgc_wait_started:
                self._wgc_wait_started = time.monotonic()
            elif time.monotonic() - self._wgc_wait_started > HOST_START_TIMEOUT:
                host.stop()
                reader.close()
                self._wgc_wait_started = 0.0
            if delay:
                time.sleep(min(delay, 0.05))
            return None

        self._wgc_wait_started = 0.0
        self.capture_backend = "Windows Graphics Capture"
        self.last_error = None
        return frame

    def patched_capture_and_track(self):
        ok = original_capture_and_track(self)
        if ok:
            now = time.monotonic()
            times = self._wgc_frame_times
            times.append(now)
            while times and now - times[0] > 1.5:
                times.popleft()
            if len(times) >= 2:
                elapsed = times[-1] - times[0]
                self.fps = (len(times) - 1) / elapsed if elapsed > 0 else 0.0
            else:
                self.fps = 0.0
        return ok

    capture_class.__init__ = patched_init
    capture_class.stop = patched_stop
    capture_class.screenshot = patched_screenshot
    capture_class._capture_and_track = patched_capture_and_track
    capture_class._wgc_capture_patch_installed = True
