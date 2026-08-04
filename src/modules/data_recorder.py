"""Observation-only dataset recorder for manual gameplay demonstrations.

F8 starts a session and F9 stops it through Listener. The recorder never sends
input. It samples the latest WGC frame, currently pressed keys, minimap position,
and SceneObserver output into a timestamped dataset directory.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import keyboard as kb

from src.common import config


@dataclass(frozen=True)
class RecorderSnapshot:
    recording: bool = False
    session_name: str = ""
    elapsed_seconds: float = 0.0
    samples: int = 0
    frames_saved: int = 0
    events: int = 0
    output_directory: str = ""
    last_event: str = "Idle"
    last_error: Optional[str] = None


class DataRecorder:
    """Record synchronized frames, actions, trajectory, and scene metadata."""

    SAMPLE_FPS = 10
    FRAME_FPS = 5
    JPEG_QUALITY = 88
    SAMPLE_INTERVAL = 1.0 / SAMPLE_FPS
    FRAME_INTERVAL = 1.0 / FRAME_FPS
    TRACKED_KEYS = (
        "left", "right", "up", "down", "space", "ctrl", "shift", "alt",
        "z", "x", "c", "a", "s", "d", "q", "w", "e", "r",
        "1", "2", "3", "4", "5", "6", "7", "8", "9",
    )

    def __init__(self):
        config.data_recorder = self
        self.ready = False
        self.recording = False
        self.last_error: Optional[str] = None
        self._session_dir: Optional[Path] = None
        self._frames_dir: Optional[Path] = None
        self._started_monotonic = 0.0
        self._started_wall = ""
        self._last_frame_save = 0.0
        self._last_source_frame_id = -1
        self._samples = 0
        self._frames_saved = 0
        self._events = 0
        self._last_event = "Idle"
        self._last_keys: Dict[str, bool] = {}
        self._last_position: Optional[Tuple[float, float]] = None
        self._samples_file = None
        self._events_file = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._main, name="data-recorder", daemon=True)

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def start(self):
        print("\n[~] Started observation data recorder (idle)")
        self.thread.start()

    def stop(self):
        self.stop_session()
        self._stop_event.set()

    def snapshot(self) -> RecorderSnapshot:
        with self._lock:
            elapsed = time.monotonic() - self._started_monotonic if self.recording else 0.0
            return RecorderSnapshot(
                recording=self.recording,
                session_name=self._session_dir.name if self._session_dir else "",
                elapsed_seconds=max(0.0, elapsed),
                samples=self._samples,
                frames_saved=self._frames_saved,
                events=self._events,
                output_directory=str(self._session_dir or ""),
                last_event=self._last_event,
                last_error=self.last_error,
            )

    def start_session(self) -> bool:
        with self._lock:
            if self.recording:
                print("\n[!] Recorder is already running")
                return False

            capture = getattr(config, "capture", None)
            if capture is None or getattr(capture, "frame", None) is None:
                self.last_error = "No WGC frame is available"
                print("\n[!] Cannot start recording: no WGC frame is available")
                return False

            session_name = datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
            session_dir = self._project_root() / "datasets" / session_name
            frames_dir = session_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=False)

            self._session_dir = session_dir
            self._frames_dir = frames_dir
            self._samples_file = (session_dir / "samples.jsonl").open("w", encoding="utf-8")
            self._events_file = (session_dir / "events.jsonl").open("w", encoding="utf-8")
            self._started_monotonic = time.monotonic()
            self._started_wall = datetime.now().isoformat(timespec="seconds")
            self._last_frame_save = 0.0
            self._last_source_frame_id = -1
            self._samples = 0
            self._frames_saved = 0
            self._events = 0
            self._last_keys = {}
            self._last_position = None
            self._last_event = "Recording started"
            self.last_error = None
            self.recording = True

            metadata = {
                "format_version": 1,
                "started_at": self._started_wall,
                "sample_fps": self.SAMPLE_FPS,
                "frame_fps": self.FRAME_FPS,
                "jpeg_quality": self.JPEG_QUALITY,
                "tracked_keys": list(self.TRACKED_KEYS),
                "observation_only": True,
            }
            (session_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        print(f"\n[~] Observation recording started: {session_dir}")
        return True

    def stop_session(self) -> bool:
        with self._lock:
            if not self.recording:
                return False
            self.recording = False
            elapsed = max(0.0, time.monotonic() - self._started_monotonic)
            statistics = {
                "started_at": self._started_wall,
                "stopped_at": datetime.now().isoformat(timespec="seconds"),
                "duration_seconds": round(elapsed, 3),
                "samples": self._samples,
                "frames_saved": self._frames_saved,
                "events": self._events,
                "output_directory": str(self._session_dir or ""),
            }
            if self._session_dir is not None:
                (self._session_dir / "statistics.json").write_text(
                    json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            for handle in (self._samples_file, self._events_file):
                if handle is not None:
                    handle.flush()
                    handle.close()
            self._samples_file = None
            self._events_file = None
            self._last_event = "Recording stopped"
            session_dir = self._session_dir

        print(f"\n[~] Observation recording stopped: {session_dir}")
        return True

    @staticmethod
    def _pressed_keys() -> Dict[str, bool]:
        state = {}
        for key in DataRecorder.TRACKED_KEYS:
            try:
                state[key] = bool(kb.is_pressed(key))
            except Exception:
                state[key] = False
        return state

    @staticmethod
    def _scene_payload():
        observer = getattr(config, "scene_observer", None)
        if observer is None:
            return None
        scene = observer.snapshot()

        def detection(item):
            if item is None:
                return None
            return {
                "label": item.label,
                "confidence": round(float(item.confidence), 4),
                "box": list(item.box),
                "center": list(item.center),
            }

        return {
            "model_status": scene.model_status,
            "player": detection(scene.player),
            "monsters": [detection(item) for item in scene.monsters],
            "ladders": [detection(item) for item in scene.ladders],
            "obstacles": [detection(item) for item in scene.obstacles],
            "nearest_monster_direction": scene.nearest_monster_direction,
            "nearest_monster_distance": scene.nearest_monster_distance,
            "last_event": scene.last_event,
        }

    def _write_event(self, timestamp: float, event_type: str, details: Dict) -> None:
        if self._events_file is None:
            return
        payload = {"time": round(timestamp, 4), "event": event_type, "details": details}
        self._events_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._events += 1
        self._last_event = event_type

    def _record_once(self) -> None:
        capture = getattr(config, "capture", None)
        if capture is None:
            raise RuntimeError("Capture service is unavailable")

        with capture._state_lock:
            frame = None if capture.frame is None else capture.frame.copy()
            frame_id = int(getattr(capture, "frame_id", 0))
            player_found = bool(getattr(capture, "player_found", False))
            player_confidence = float(getattr(capture, "player_confidence", 0.0))

        if frame is None or frame_id == self._last_source_frame_id:
            return
        self._last_source_frame_id = frame_id
        elapsed = time.monotonic() - self._started_monotonic
        keys = self._pressed_keys()
        position = tuple(float(v) for v in getattr(config, "player_pos", (0.0, 0.0)))

        for key, pressed in keys.items():
            previous = self._last_keys.get(key, False)
            if pressed != previous:
                self._write_event(elapsed, "key_down" if pressed else "key_up", {"key": key})

        directional_pressed = any(keys.get(key, False) for key in ("left", "right", "up", "down"))
        if player_found and self._last_position is not None and not directional_pressed:
            dx = position[0] - self._last_position[0]
            dy = position[1] - self._last_position[1]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance >= 0.035:
                self._write_event(
                    elapsed,
                    "possible_knockback_or_forced_movement",
                    {"dx": round(dx, 5), "dy": round(dy, 5), "distance": round(distance, 5)},
                )

        frame_name = None
        if elapsed - self._last_frame_save >= self.FRAME_INTERVAL and self._frames_dir is not None:
            frame_name = f"frame_{self._frames_saved:07d}.jpg"
            output = self._frames_dir / frame_name
            if cv2.imwrite(str(output), frame[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY]):
                self._frames_saved += 1
                self._last_frame_save = elapsed

        sample = {
            "time": round(elapsed, 4),
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "source_frame_id": frame_id,
            "frame_file": frame_name,
            "keys": keys,
            "player": {
                "minimap_position": [round(position[0], 6), round(position[1], 6)],
                "found": player_found,
                "confidence": round(player_confidence, 4),
            },
            "scene": self._scene_payload(),
        }
        if self._samples_file is not None:
            self._samples_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if self._samples % 20 == 0:
                self._samples_file.flush()
                if self._events_file is not None:
                    self._events_file.flush()
        self._samples += 1
        self._last_keys = keys
        self._last_position = position if player_found else None

    def _main(self):
        self.ready = True
        while not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                if self.recording:
                    self._record_once()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._last_event = "Recorder error"
                print(f"\n[!] Data recorder error: {self.last_error}")
                time.sleep(0.5)

            remaining = self.SAMPLE_INTERVAL - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
