"""Immutable state objects shared safely between worker threads and the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


Point = Tuple[float, float]
WindowRect = Tuple[int, int, int, int]


@dataclass(frozen=True)
class CaptureSnapshot:
    frame_id: int = 0
    timestamp: float = 0.0
    fps: float = 0.0
    window_found: bool = False
    calibrated: bool = False
    player_found: bool = False
    player_position: Point = (0.0, 0.0)
    player_confidence: float = 0.0
    window_rect: WindowRect = (0, 0, 0, 0)
    rejected_player_jumps: int = 0
    last_error: Optional[str] = None

    @property
    def healthy(self) -> bool:
        """Whether the capture state is suitable for read-only consumers."""
        return self.window_found and self.calibrated and self.player_found and not self.last_error


@dataclass(frozen=True)
class RuntimeHealthSnapshot:
    timestamp: float
    enabled: bool
    capture_thread_alive: bool
    watchdog_thread_alive: bool
    capture_frame_age: Optional[float]
    capture: CaptureSnapshot
    status: str
