"""Runtime health aggregation for diagnostics and future GUI display."""

from __future__ import annotations

import time
from typing import Any

from src.common import config
from src.common.snapshots import CaptureSnapshot, RuntimeHealthSnapshot


def _capture_snapshot(capture: Any) -> CaptureSnapshot:
    """Convert a capture object into a stable immutable snapshot."""
    if capture is None:
        return CaptureSnapshot(last_error="Capture service is unavailable")

    health = capture.health_snapshot() if hasattr(capture, "health_snapshot") else {}
    window = getattr(capture, "window", {}) or {}
    player_position = tuple(getattr(config, "player_pos", (0.0, 0.0)))

    return CaptureSnapshot(
        frame_id=int(health.get("frame_id", 0)),
        timestamp=float(getattr(capture, "last_frame_time", 0.0)),
        fps=float(getattr(capture, "fps", 0.0)),
        window_found=bool(health.get("window_found", False)),
        calibrated=bool(health.get("calibrated", False)),
        player_found=bool(health.get("player_found", False)),
        player_position=(float(player_position[0]), float(player_position[1])),
        player_confidence=float(health.get("player_confidence", 0.0)),
        window_rect=(
            int(window.get("left", 0)),
            int(window.get("top", 0)),
            int(window.get("width", 0)),
            int(window.get("height", 0)),
        ),
        rejected_player_jumps=int(health.get("rejected_player_jumps", 0)),
        last_error=health.get("last_error"),
    )


def runtime_health() -> RuntimeHealthSnapshot:
    """Return a read-only health summary without mutating application state."""
    now = time.monotonic()
    capture = getattr(config, "capture", None)
    snapshot = _capture_snapshot(capture)

    capture_thread = getattr(capture, "thread", None)
    watchdog_thread = getattr(capture, "watchdog_thread", None)
    frame_age = None
    if snapshot.timestamp:
        frame_age = max(0.0, now - snapshot.timestamp)

    capture_alive = bool(capture_thread and capture_thread.is_alive())
    watchdog_alive = bool(watchdog_thread and watchdog_thread.is_alive())

    if snapshot.last_error:
        status = "error"
    elif not capture_alive:
        status = "capture-stopped"
    elif not snapshot.window_found:
        status = "waiting-for-window"
    elif not snapshot.calibrated:
        status = "calibrating"
    elif not snapshot.player_found:
        status = "waiting-for-player"
    else:
        status = "healthy"

    return RuntimeHealthSnapshot(
        timestamp=now,
        enabled=bool(getattr(config, "enabled", False)),
        capture_thread_alive=capture_alive,
        watchdog_thread_alive=watchdog_alive,
        capture_frame_age=frame_age,
        capture=snapshot,
        status=status,
    )
