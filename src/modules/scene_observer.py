"""Non-blocking main-scene observation for the classic client.

This phase is observation-only: it never presses game keys. It reads the latest WGC
frame, runs the trained classic_scene.pt detector, publishes diagnostics for the GUI,
and can show a live F10 debug preview with YOLO boxes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.common import config

Box = Tuple[int, int, int, int]
Point = Tuple[int, int]


@dataclass(frozen=True)
class SceneDetection:
    label: str
    confidence: float
    box: Box

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass(frozen=True)
class SceneSnapshot:
    timestamp: float = 0.0
    frame_id: int = 0
    mode: str = "observe"
    model_status: str = "not-loaded"
    inference_ms: float = 0.0
    player: Optional[SceneDetection] = None
    monsters: Tuple[SceneDetection, ...] = field(default_factory=tuple)
    ladders: Tuple[SceneDetection, ...] = field(default_factory=tuple)
    obstacles: Tuple[SceneDetection, ...] = field(default_factory=tuple)
    nearest_monster_direction: str = "none"
    nearest_monster_distance: Optional[float] = None
    last_event: str = "Waiting for frame"
    last_error: Optional[str] = None


class SceneObserver:
    """Observe the gameplay scene without delaying capture or controlling input."""

    TARGET_FPS = 10
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    MODEL_FILENAME = "classic_scene.pt"
    DEBUG_WINDOW = "Auto Maple AI - Monster Vision (F10 to close)"
    CLASS_ALIASES: Dict[str, str] = {
        "character": "player",
        "hero": "player",
        "mob": "monster",
        "enemy": "monster",
        "rope": "ladder",
        "wall": "obstacle",
        "block": "obstacle",
    }

    def __init__(self):
        config.scene_observer = self
        self.ready = False
        self.enabled = True
        self.debug_enabled = False
        self.last_error: Optional[str] = None
        self._model = None
        self._model_attempted = False
        self._model_status = "not-loaded"
        self._last_source_frame_id = -1
        self._snapshot = SceneSnapshot()
        self._debug_frame: Optional[np.ndarray] = None
        self._debug_window_open = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._main,
            name="scene-observer",
            daemon=True,
        )

    def start(self):
        print("\n[~] Started main-scene observer (observation only)")
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        self._close_debug_window()

    def snapshot(self) -> SceneSnapshot:
        with self._lock:
            return self._snapshot

    def debug_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._debug_frame is None else self._debug_frame.copy()

    def toggle_debug(self) -> bool:
        self.debug_enabled = not self.debug_enabled
        if not self.debug_enabled:
            with self._lock:
                self._debug_frame = None
            self._close_debug_window()
        print(f"\n[~] Vision debug {'enabled' if self.debug_enabled else 'disabled'}")
        if self.debug_enabled:
            print("[~] Live monster preview will open when the next WGC frame is processed")
        return self.debug_enabled

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _load_model(self):
        if self._model_attempted:
            return self._model
        self._model_attempted = True
        weights = self._project_root() / "assets" / "models" / self.MODEL_FILENAME
        if not weights.is_file():
            self._model_status = "model-not-trained"
            return None
        try:
            from ultralytics import YOLO

            print(f"\n[~] Loading scene model: {weights}")
            self._model = YOLO(str(weights))
            self._model_status = "ready"
            print("[~] Scene model ready: classic_scene.pt")
        except Exception as exc:
            self._model_status = "load-error"
            self.last_error = f"Scene model load failed: {exc}"
            self._model = None
        return self._model

    @staticmethod
    def _gameplay_bounds(frame: np.ndarray, capture) -> Box:
        height, width = frame.shape[:2]
        snapshot = getattr(capture, "fixed_ui_snapshot", None)
        if snapshot is not None:
            region = getattr(snapshot, "gameplay_area", None)
            if region is not None:
                (x1, y1), (x2, y2) = region.bounds
                return max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        return 0, int(height * 0.08), width, int(height * 0.855)

    @classmethod
    def _canonical_label(cls, label: str) -> str:
        normalized = label.casefold().strip()
        return cls.CLASS_ALIASES.get(normalized, normalized)

    def _predict(self, frame: np.ndarray, bounds: Box) -> Tuple[List[SceneDetection], float]:
        model = self._load_model()
        if model is None:
            return [], 0.0
        x1, y1, x2, y2 = bounds
        crop = frame[y1:y2, x1:x2, :3]
        if crop.size == 0:
            return [], 0.0

        started = time.perf_counter()
        results = model.predict(crop, imgsz=640, conf=0.45, verbose=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        detections: List[SceneDetection] = []
        for result in results:
            names = result.names
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                label = self._canonical_label(str(names.get(class_id, class_id)))
                confidence = float(box.conf[0])
                bx1, by1, bx2, by2 = (
                    int(round(value)) for value in box.xyxy[0].tolist()
                )
                detections.append(
                    SceneDetection(
                        label=label,
                        confidence=confidence,
                        box=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                    )
                )
        return detections, elapsed_ms

    @staticmethod
    def _nearest(player: Optional[SceneDetection], monsters: List[SceneDetection]):
        if player is None or not monsters:
            return "none", None
        px, py = player.center
        nearest = min(
            monsters,
            key=lambda item: (item.center[0] - px) ** 2 + (item.center[1] - py) ** 2,
        )
        mx, my = nearest.center
        dx, dy = mx - px, my - py
        distance = float((dx * dx + dy * dy) ** 0.5)
        if abs(dx) >= abs(dy):
            direction = "right" if dx >= 0 else "left"
        else:
            direction = "below" if dy >= 0 else "above"
        return direction, distance

    @staticmethod
    def _draw_debug(
        frame: np.ndarray,
        detections: List[SceneDetection],
        bounds: Box,
        inference_ms: float,
        model_status: str,
    ) -> np.ndarray:
        debug = frame[:, :, :3].copy()
        x1, y1, x2, y2 = bounds
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 255), 1)

        monsters = 0
        for item in detections:
            bx1, by1, bx2, by2 = item.box
            if item.label == "monster":
                monsters += 1
            cv2.rectangle(debug, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
            cv2.putText(
                debug,
                f"{item.label} {item.confidence * 100:.0f}%",
                (bx1, max(20, by1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        hud = f"AI: {model_status}   Monsters: {monsters}   Inference: {inference_ms:.1f} ms"
        cv2.rectangle(debug, (8, 8), (min(debug.shape[1] - 8, 590), 42), (0, 0, 0), -1)
        cv2.putText(
            debug,
            hud,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return debug

    def _show_debug_window(self, debug: np.ndarray) -> None:
        try:
            if not self._debug_window_open:
                cv2.namedWindow(self.DEBUG_WINDOW, cv2.WINDOW_NORMAL)
                height, width = debug.shape[:2]
                target_width = min(1280, width)
                target_height = max(360, int(height * (target_width / max(1, width))))
                cv2.resizeWindow(self.DEBUG_WINDOW, target_width, target_height)
                self._debug_window_open = True
            cv2.imshow(self.DEBUG_WINDOW, debug)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.last_error = f"Vision preview error: {exc}"
            self._debug_window_open = False

    def _close_debug_window(self) -> None:
        if not self._debug_window_open:
            return
        try:
            cv2.destroyWindow(self.DEBUG_WINDOW)
            cv2.waitKey(1)
        except cv2.error:
            pass
        self._debug_window_open = False

    def _observe_once(self) -> None:
        capture = getattr(config, "capture", None)
        if capture is None:
            raise RuntimeError("Capture service is unavailable")

        with capture._state_lock:
            frame = None if capture.frame is None else capture.frame.copy()
            source_frame_id = int(getattr(capture, "frame_id", 0))

        if frame is None or source_frame_id == self._last_source_frame_id:
            return
        self._last_source_frame_id = source_frame_id
        bounds = self._gameplay_bounds(frame, capture)
        detections, inference_ms = self._predict(frame, bounds)

        players = [item for item in detections if item.label == "player"]
        monsters = [item for item in detections if item.label == "monster"]
        ladders = [item for item in detections if item.label == "ladder"]
        obstacles = [item for item in detections if item.label == "obstacle"]
        player = max(players, key=lambda item: item.confidence) if players else None
        direction, distance = self._nearest(player, monsters)

        if self._model_status == "model-not-trained":
            event = "Scene model not trained; capture pipeline is ready"
        elif self._model_status != "ready":
            event = f"Scene model status: {self._model_status}"
        elif monsters:
            event = f"Tracking {len(monsters)} monster(s)"
        else:
            event = "No monster detected"

        snapshot = SceneSnapshot(
            timestamp=time.monotonic(),
            frame_id=source_frame_id,
            mode="observe",
            model_status=self._model_status,
            inference_ms=inference_ms,
            player=player,
            monsters=tuple(monsters),
            ladders=tuple(ladders),
            obstacles=tuple(obstacles),
            nearest_monster_direction=direction,
            nearest_monster_distance=distance,
            last_event=event,
            last_error=self.last_error,
        )

        debug = None
        if self.debug_enabled:
            debug = self._draw_debug(frame, detections, bounds, inference_ms, self._model_status)
            self._show_debug_window(debug)
        elif self._debug_window_open:
            self._close_debug_window()

        with self._lock:
            self._snapshot = snapshot
            self._debug_frame = debug

    def _main(self):
        self.ready = True
        while not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                if self.enabled:
                    self._observe_once()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._snapshot = SceneSnapshot(
                        timestamp=time.monotonic(),
                        frame_id=self._last_source_frame_id,
                        mode="observe",
                        model_status=self._model_status,
                        last_event="Observer error",
                        last_error=self.last_error,
                    )
                time.sleep(0.5)

            remaining = self.FRAME_INTERVAL - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)

        self._close_debug_window()
