"""Low-latency combined main-scene observation for the classic client.

The production observer consumes frames already published by Windows Graphics
Capture.  It keeps the proven monster detector and navigation detector separate,
but schedules them at different rates:

* classic_scene.pt:      high-frequency monster detection
* navigation_scene.pt:  low-frequency ladder/platform detection

Ladders and platforms are effectively static scene geometry, so their most recent
results are cached between navigation passes.  F10 only visualizes the cached
production results; enabling the preview does not create another inference path.
This module never sends game input.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

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

    @property
    def foot(self) -> Point:
        x1, _y1, x2, y2 = self.box
        return ((x1 + x2) // 2, y2)


@dataclass(frozen=True)
class SceneSnapshot:
    timestamp: float = 0.0
    frame_id: int = 0
    mode: str = "observe"
    model_status: str = "not-loaded"
    inference_ms: float = 0.0
    monster_inference_ms: float = 0.0
    navigation_inference_ms: float = 0.0
    player: Optional[SceneDetection] = None
    monsters: Tuple[SceneDetection, ...] = field(default_factory=tuple)
    ladders: Tuple[SceneDetection, ...] = field(default_factory=tuple)
    platforms: Tuple[SceneDetection, ...] = field(default_factory=tuple)
    obstacles: Tuple[SceneDetection, ...] = field(default_factory=tuple)
    nearest_monster_direction: str = "none"
    nearest_monster_distance: Optional[float] = None
    last_event: str = "Waiting for frame"
    last_error: Optional[str] = None


class SceneObserver:
    """Observe the gameplay scene without delaying WGC or controlling input."""

    # Scheduler. Capture itself runs independently at ~30 FPS.
    LOOP_HZ = 30
    LOOP_INTERVAL = 1.0 / LOOP_HZ
    MONSTER_HZ = 20
    MONSTER_INTERVAL = 1.0 / MONSTER_HZ
    NAVIGATION_HZ = 4
    NAVIGATION_INTERVAL = 1.0 / NAVIGATION_HZ
    DEBUG_HZ = 10
    DEBUG_INTERVAL = 1.0 / DEBUG_HZ

    MONSTER_MODEL_FILENAME = "classic_scene.pt"
    NAVIGATION_MODEL_FILENAME = "navigation_scene.pt"
    MONSTER_CONFIDENCE = 0.45
    NAVIGATION_CONFIDENCE = 0.40
    IMAGE_SIZE = 640
    DEBUG_WINDOW = "Auto Maple AI - Combined Vision (F10 to close)"

    def __init__(self):
        config.scene_observer = self
        self.ready = False
        self.enabled = True
        self.debug_enabled = False
        self.last_error: Optional[str] = None

        self._monster_model = None
        self._navigation_model = None
        self._models_attempted = False
        self._monster_status = "not-loaded"
        self._navigation_status = "not-loaded"
        self._device = None
        self._gpu_name = "unknown"

        self._last_source_frame_id = -1
        self._last_monster_run = 0.0
        self._last_navigation_run = 0.0
        self._last_debug_draw = 0.0
        self._monster_ms = 0.0
        self._navigation_ms = 0.0
        self._monsters: List[SceneDetection] = []
        self._ladders: List[SceneDetection] = []
        self._platforms: List[SceneDetection] = []

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
        print("\n[~] Started combined main-scene observer (observation only)")
        print(f"[~] Monster vision target: {self.MONSTER_HZ} Hz")
        print(f"[~] Navigation vision target: {self.NAVIGATION_HZ} Hz")
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
        print(f"\n[~] Combined vision debug {'enabled' if self.debug_enabled else 'disabled'}")
        if self.debug_enabled:
            print("[~] F10 preview uses the production Monster + Ladder + Platform cache")
        return self.debug_enabled

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _load_models(self) -> None:
        if self._models_attempted:
            return
        self._models_attempted = True

        root = self._project_root()
        monster_weights = root / "assets" / "models" / self.MONSTER_MODEL_FILENAME
        nav_weights = root / "assets" / "models" / self.NAVIGATION_MODEL_FILENAME

        try:
            import torch
            from ultralytics import YOLO
        except Exception as exc:
            self.last_error = f"YOLO/PyTorch unavailable: {exc}"
            self._monster_status = "load-error"
            self._navigation_status = "load-error"
            return

        if torch.cuda.is_available():
            self._device = "0"
            self._gpu_name = torch.cuda.get_device_name(0)
        else:
            self._device = "cpu"
            self._gpu_name = "CPU"

        if monster_weights.is_file():
            try:
                print(f"\n[~] Loading monster model: {monster_weights}")
                self._monster_model = YOLO(str(monster_weights))
                self._monster_status = "ready"
                print("[~] Monster model ready: classic_scene.pt")
            except Exception as exc:
                self._monster_status = "load-error"
                self.last_error = f"Monster model load failed: {exc}"
        else:
            self._monster_status = "model-not-trained"

        if nav_weights.is_file():
            try:
                print(f"\n[~] Loading navigation model: {nav_weights}")
                self._navigation_model = YOLO(str(nav_weights))
                self._navigation_status = "ready"
                print("[~] Navigation model ready: navigation_scene.pt")
            except Exception as exc:
                self._navigation_status = "load-error"
                self.last_error = f"Navigation model load failed: {exc}"
        else:
            self._navigation_status = "model-not-trained"

        print(f"[~] Vision device: {self._gpu_name} ({self._device})")

    @staticmethod
    def _full_frame_bounds(frame: np.ndarray) -> Box:
        # Combined test and training images use the full WGC frame. Keeping the
        # production observer identical avoids crop-induced misses near map edges.
        height, width = frame.shape[:2]
        return 0, 0, width, height

    @staticmethod
    def _predict_model(model, frame: np.ndarray, confidence: float, device) -> Tuple[object, float]:
        # WGC publishes BGRA. Ultralytics/OpenCV numpy input is BGR.
        bgr = frame[:, :, :3]
        started = time.perf_counter()
        results = model.predict(
            bgr,
            imgsz=SceneObserver.IMAGE_SIZE,
            conf=confidence,
            device=device,
            verbose=False,
        )
        return results, (time.perf_counter() - started) * 1000.0

    @staticmethod
    def _monster_detections(results) -> List[SceneDetection]:
        detections: List[SceneDetection] = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                label = str(names.get(class_id, class_id)).casefold().strip()
                if label not in {"monster", "mob", "enemy", "0"} and len(names) > 1:
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                detections.append(
                    SceneDetection("monster", float(box.conf[0]), (x1, y1, x2, y2))
                )
        return detections

    @staticmethod
    def _navigation_detections(results) -> Tuple[List[SceneDetection], List[SceneDetection]]:
        ladders: List[SceneDetection] = []
        platforms: List[SceneDetection] = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                label = str(names.get(class_id, class_id)).casefold().strip()
                x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                detection = SceneDetection(label, float(box.conf[0]), (x1, y1, x2, y2))
                if label == "ladder" or (len(names) == 2 and class_id == 0):
                    ladders.append(SceneDetection("ladder", detection.confidence, detection.box))
                elif label == "platform" or (len(names) == 2 and class_id == 1):
                    platforms.append(SceneDetection("platform", detection.confidence, detection.box))
        return ladders, platforms

    @staticmethod
    def _center_anchor(frame: np.ndarray) -> Point:
        # The scrolling camera normally keeps the local character near center.
        # This is only a relative combat anchor; minimap remains the global locator.
        height, width = frame.shape[:2]
        return width // 2, height // 2

    @classmethod
    def _nearest_to_anchor(cls, frame: np.ndarray, monsters: List[SceneDetection]):
        if not monsters:
            return "none", None
        px, py = cls._center_anchor(frame)
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

    def _model_status_text(self) -> str:
        return f"monster={self._monster_status}, nav={self._navigation_status}"

    def _draw_debug(self, frame: np.ndarray) -> np.ndarray:
        debug = frame[:, :, :3].copy()
        colors = {
            "monster": (20, 255, 57),
            "ladder": (0, 215, 255),
            "platform": (80, 127, 255),
        }
        for item in [*self._monsters, *self._ladders, *self._platforms]:
            x1, y1, x2, y2 = item.box
            color = colors.get(item.label, (255, 255, 255))
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                debug,
                f"{item.label} {item.confidence * 100:.0f}%",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                1,
                cv2.LINE_AA,
            )

        px, py = self._center_anchor(debug)
        cv2.drawMarker(debug, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

        hud1 = (
            f"GPU: {self._gpu_name} | Monsters: {len(self._monsters)} | "
            f"Ladders: {len(self._ladders)} | Platforms: {len(self._platforms)}"
        )
        hud2 = (
            f"Monster: {self._monster_ms:.1f} ms @ {self.MONSTER_HZ}Hz | "
            f"Nav: {self._navigation_ms:.1f} ms @ {self.NAVIGATION_HZ}Hz | "
            f"Preview: {self.DEBUG_HZ}Hz"
        )
        cv2.rectangle(debug, (8, 8), (min(debug.shape[1] - 8, 980), 66), (0, 0, 0), -1)
        cv2.putText(debug, hud1, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(debug, hud2, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
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

        self._load_models()

        with capture._state_lock:
            frame = None if capture.frame is None else capture.frame.copy()
            source_frame_id = int(getattr(capture, "frame_id", 0))

        if frame is None or source_frame_id == self._last_source_frame_id:
            return
        self._last_source_frame_id = source_frame_id
        now = time.monotonic()

        if self._monster_model is not None and now - self._last_monster_run >= self.MONSTER_INTERVAL:
            results, self._monster_ms = self._predict_model(
                self._monster_model,
                frame,
                self.MONSTER_CONFIDENCE,
                self._device,
            )
            self._monsters = self._monster_detections(results)
            self._last_monster_run = now

        if self._navigation_model is not None and now - self._last_navigation_run >= self.NAVIGATION_INTERVAL:
            results, self._navigation_ms = self._predict_model(
                self._navigation_model,
                frame,
                self.NAVIGATION_CONFIDENCE,
                self._device,
            )
            self._ladders, self._platforms = self._navigation_detections(results)
            self._last_navigation_run = now

        direction, distance = self._nearest_to_anchor(frame, self._monsters)
        total_ms = self._monster_ms + self._navigation_ms

        if self._monster_status != "ready" and self._navigation_status != "ready":
            event = "Scene models are not ready"
        elif self._monsters:
            event = (
                f"Tracking {len(self._monsters)} monster(s), "
                f"{len(self._ladders)} ladder(s), {len(self._platforms)} platform(s)"
            )
        else:
            event = f"No monster detected; navigation cache has {len(self._ladders)} ladder(s)"

        snapshot = SceneSnapshot(
            timestamp=now,
            frame_id=source_frame_id,
            mode="observe-combined",
            model_status=self._model_status_text(),
            inference_ms=total_ms,
            monster_inference_ms=self._monster_ms,
            navigation_inference_ms=self._navigation_ms,
            monsters=tuple(self._monsters),
            ladders=tuple(self._ladders),
            platforms=tuple(self._platforms),
            nearest_monster_direction=direction,
            nearest_monster_distance=distance,
            last_event=event,
            last_error=self.last_error,
        )

        debug = None
        if self.debug_enabled and now - self._last_debug_draw >= self.DEBUG_INTERVAL:
            debug = self._draw_debug(frame)
            self._show_debug_window(debug)
            self._last_debug_draw = now
        elif not self.debug_enabled and self._debug_window_open:
            self._close_debug_window()

        with self._lock:
            self._snapshot = snapshot
            if debug is not None:
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
                        mode="observe-combined",
                        model_status=self._model_status_text(),
                        last_event="Observer error",
                        last_error=self.last_error,
                    )
                time.sleep(0.25)

            remaining = self.LOOP_INTERVAL - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)

        self._close_debug_window()
