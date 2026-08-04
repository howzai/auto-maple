"""Smart observation-only dataset recorder for manual gameplay demonstrations.

F8 starts a session and F9 stops it. The recorder never sends game input. It
samples actions and minimap state at 10 Hz while saving images adaptively: active
combat/movement receives dense coverage, quiet or duplicate scenes are sampled
sparsely. Each completed session receives quality and AI-readiness reports.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import keyboard as kb
import numpy as np

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
    recording_mode: str = "balanced-smart"
    estimated_size_mb: float = 0.0
    quality_score: int = 0
    readiness_score: int = 0
    recommendation: str = "Not evaluated"
    last_event: str = "Idle"
    last_error: Optional[str] = None


class DataRecorder:
    """Record synchronized frames, actions, trajectory, and scene metadata."""

    SAMPLE_FPS = 10
    SAMPLE_INTERVAL = 1.0 / SAMPLE_FPS
    RECORDING_MODE = "balanced-smart"
    JPEG_QUALITY = 82

    # Smart image cadence. JSON samples remain fixed at 10 Hz.
    EVENT_FRAME_INTERVAL = 0.125      # 8 FPS during input/events
    MOTION_FRAME_INTERVAL = 0.25     # 4 FPS while the scene changes
    BASE_FRAME_INTERVAL = 0.50       # 2 FPS during ordinary play
    IDLE_FRAME_INTERVAL = 1.50       # sparse coverage when nearly static
    EVENT_BURST_SECONDS = 0.80
    SCENE_CHANGE_THRESHOLD = 5.5
    IDLE_CHANGE_THRESHOLD = 1.25

    TRACKED_KEYS = (
        "left", "right", "up", "down", "space", "ctrl", "shift", "alt",
        "z", "x", "c", "a", "s", "d", "q", "w", "e", "r",
        "1", "2", "3", "4", "5", "6", "7", "8", "9",
    )
    MOVEMENT_KEYS = {"left", "right", "up", "down", "space"}
    ACTION_KEYS = {"ctrl", "shift", "alt", "z", "x", "c", "a", "s", "d",
                   "q", "w", "e", "r", "1", "2", "3", "4", "5", "6",
                   "7", "8", "9"}

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
        self._last_event_time = -999.0
        self._last_preview: Optional[np.ndarray] = None
        self._last_bounds = None
        self._samples = 0
        self._frames_saved = 0
        self._events = 0
        self._valid_player_samples = 0
        self._calibrated_samples = 0
        self._black_frames = 0
        self._duplicate_skips = 0
        self._roi_changes = 0
        self._active_samples = 0
        self._scene_model_ready_samples = 0
        self._monster_samples = 0
        self._ladder_samples = 0
        self._event_counts: Counter = Counter()
        self._key_down_counts: Counter = Counter()
        self._last_event = "Idle"
        self._last_keys: Dict[str, bool] = {}
        self._last_position: Optional[Tuple[float, float]] = None
        self._samples_file = None
        self._events_file = None
        self._last_report: Dict = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._main, name="data-recorder", daemon=True)

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def start(self):
        print("\n[~] Started smart observation data recorder (idle)")
        self.thread.start()

    def stop(self):
        self.stop_session()
        self._stop_event.set()

    def _directory_size_mb(self) -> float:
        directory = self._session_dir
        if directory is None or not directory.exists():
            return 0.0
        try:
            return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()) / 1048576.0
        except OSError:
            return 0.0

    def snapshot(self) -> RecorderSnapshot:
        with self._lock:
            elapsed = time.monotonic() - self._started_monotonic if self.recording else 0.0
            report = self._last_report
            return RecorderSnapshot(
                recording=self.recording,
                session_name=self._session_dir.name if self._session_dir else "",
                elapsed_seconds=max(0.0, elapsed),
                samples=self._samples,
                frames_saved=self._frames_saved,
                events=self._events,
                output_directory=str(self._session_dir or ""),
                recording_mode=self.RECORDING_MODE,
                estimated_size_mb=round(self._directory_size_mb(), 1),
                quality_score=int(report.get("quality_score", 0)),
                readiness_score=int(report.get("overall_readiness", 0)),
                recommendation=str(report.get("recommendation", "Recording" if self.recording else "Not evaluated")),
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
            self._last_event_time = -999.0
            self._last_preview = None
            self._last_bounds = None
            self._samples = self._frames_saved = self._events = 0
            self._valid_player_samples = self._calibrated_samples = 0
            self._black_frames = self._duplicate_skips = self._roi_changes = 0
            self._active_samples = self._scene_model_ready_samples = 0
            self._monster_samples = self._ladder_samples = 0
            self._event_counts = Counter()
            self._key_down_counts = Counter()
            self._last_keys = {}
            self._last_position = None
            self._last_report = {}
            self._last_event = "Recording started"
            self.last_error = None
            self.recording = True

            metadata = {
                "format_version": 2,
                "started_at": self._started_wall,
                "sample_fps": self.SAMPLE_FPS,
                "recording_mode": self.RECORDING_MODE,
                "jpeg_quality": self.JPEG_QUALITY,
                "frame_intervals": {
                    "event": self.EVENT_FRAME_INTERVAL,
                    "motion": self.MOTION_FRAME_INTERVAL,
                    "base": self.BASE_FRAME_INTERVAL,
                    "idle": self.IDLE_FRAME_INTERVAL,
                },
                "tracked_keys": list(self.TRACKED_KEYS),
                "observation_only": True,
            }
            (session_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        print(f"\n[~] Smart observation recording started: {session_dir}")
        return True

    def stop_session(self) -> bool:
        with self._lock:
            if not self.recording:
                return False
            self.recording = False
            elapsed = max(0.0, time.monotonic() - self._started_monotonic)
            for handle in (self._samples_file, self._events_file):
                if handle is not None:
                    handle.flush()
                    handle.close()
            self._samples_file = None
            self._events_file = None
            report = self._build_readiness_report(elapsed)
            self._last_report = report
            session_dir = self._session_dir
            if session_dir is not None:
                statistics = {
                    "started_at": self._started_wall,
                    "stopped_at": datetime.now().isoformat(timespec="seconds"),
                    "duration_seconds": round(elapsed, 3),
                    "samples": self._samples,
                    "frames_saved": self._frames_saved,
                    "events": self._events,
                    "size_mb": round(self._directory_size_mb(), 2),
                    "event_counts": dict(self._event_counts),
                    "key_down_counts": dict(self._key_down_counts),
                    "output_directory": str(session_dir),
                }
                (session_dir / "statistics.json").write_text(
                    json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (session_dir / "readiness_report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (session_dir / "readiness_report.txt").write_text(
                    self._human_report(report), encoding="utf-8"
                )
            self._last_event = "Recording stopped"

        stars = "★" * int(report["stars"]) + "☆" * (5 - int(report["stars"]))
        print(f"\n[~] Smart observation recording stopped: {session_dir}")
        print(f"[~] Dataset quality: {stars} {report['quality_score']}%")
        print(f"[~] AI readiness: {report['overall_readiness']}% · {report['recommendation']}")
        for suggestion in report["suggestions"][:4]:
            print(f"    - {suggestion}")
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
            return {"label": item.label, "confidence": round(float(item.confidence), 4),
                    "box": list(item.box), "center": list(item.center)}

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
        self._event_counts[event_type] += 1
        self._last_event = event_type
        self._last_event_time = timestamp

    @staticmethod
    def _preview(frame: np.ndarray) -> np.ndarray:
        bgr = frame[:, :, :3]
        height, width = bgr.shape[:2]
        scale = min(1.0, 240.0 / max(width, 1))
        resized = cv2.resize(bgr, (max(1, int(width * scale)), max(1, int(height * scale))))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def _scene_change(self, frame: np.ndarray) -> float:
        preview = self._preview(frame)
        if self._last_preview is None or self._last_preview.shape != preview.shape:
            self._last_preview = preview
            return 255.0
        difference = float(np.mean(cv2.absdiff(preview, self._last_preview)))
        self._last_preview = preview
        return difference

    @staticmethod
    def _is_black_frame(frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < 5.0 or float(np.std(gray)) < 2.0

    def _frame_interval(self, elapsed: float, keys: Dict[str, bool], scene_change: float,
                        forced_movement: bool, scene: Optional[Dict]) -> float:
        active_key = any(keys.values())
        detected_objects = bool(scene and (scene.get("monsters") or scene.get("ladders")))
        in_event_burst = elapsed - self._last_event_time <= self.EVENT_BURST_SECONDS
        if active_key or forced_movement or detected_objects or in_event_burst:
            return self.EVENT_FRAME_INTERVAL
        if scene_change >= self.SCENE_CHANGE_THRESHOLD:
            return self.MOTION_FRAME_INTERVAL
        if scene_change <= self.IDLE_CHANGE_THRESHOLD:
            return self.IDLE_FRAME_INTERVAL
        return self.BASE_FRAME_INTERVAL

    def _record_once(self) -> None:
        capture = getattr(config, "capture", None)
        if capture is None:
            raise RuntimeError("Capture service is unavailable")
        with capture._state_lock:
            frame = None if capture.frame is None else capture.frame.copy()
            frame_id = int(getattr(capture, "frame_id", 0))
            player_found = bool(getattr(capture, "player_found", False))
            player_confidence = float(getattr(capture, "player_confidence", 0.0))
            calibrated = bool(getattr(capture, "calibrated", False))
            bounds = (tuple(getattr(capture, "_minimap_tl", (0, 0))),
                      tuple(getattr(capture, "_minimap_br", (0, 0))))

        if frame is None or frame_id == self._last_source_frame_id:
            return
        self._last_source_frame_id = frame_id
        elapsed = time.monotonic() - self._started_monotonic
        keys = self._pressed_keys()
        position = tuple(float(v) for v in getattr(config, "player_pos", (0.0, 0.0)))
        scene = self._scene_payload()

        if calibrated:
            self._calibrated_samples += 1
        if player_found:
            self._valid_player_samples += 1
        if any(keys.values()):
            self._active_samples += 1
        if scene and scene.get("model_status") == "ready":
            self._scene_model_ready_samples += 1
            self._monster_samples += len(scene.get("monsters") or [])
            self._ladder_samples += len(scene.get("ladders") or [])

        if self._last_bounds is not None and bounds != self._last_bounds:
            self._roi_changes += 1
            self._write_event(elapsed, "minimap_roi_changed", {"old": self._last_bounds, "new": bounds})
        self._last_bounds = bounds

        for key, pressed in keys.items():
            previous = self._last_keys.get(key, False)
            if pressed != previous:
                event_type = "key_down" if pressed else "key_up"
                self._write_event(elapsed, event_type, {"key": key})
                if pressed:
                    self._key_down_counts[key] += 1

        directional_pressed = any(keys.get(key, False) for key in ("left", "right", "up", "down"))
        forced_movement = False
        if player_found and self._last_position is not None and not directional_pressed:
            dx = position[0] - self._last_position[0]
            dy = position[1] - self._last_position[1]
            distance = (dx * dx + dy * dy) ** 0.5
            if distance >= 0.035:
                forced_movement = True
                self._write_event(elapsed, "possible_knockback_or_forced_movement",
                                  {"dx": round(dx, 5), "dy": round(dy, 5),
                                   "distance": round(distance, 5)})

        scene_change = self._scene_change(frame)
        frame_name = None
        interval = self._frame_interval(elapsed, keys, scene_change, forced_movement, scene)
        should_save = elapsed - self._last_frame_save >= interval
        if should_save and self._frames_dir is not None:
            if self._is_black_frame(frame):
                self._black_frames += 1
            else:
                frame_name = f"frame_{self._frames_saved:07d}.jpg"
                output = self._frames_dir / frame_name
                ok = cv2.imwrite(str(output), frame[:, :, :3],
                                 [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
                if ok:
                    self._frames_saved += 1
                    self._last_frame_save = elapsed
        elif scene_change <= self.IDLE_CHANGE_THRESHOLD:
            self._duplicate_skips += 1

        sample = {
            "time": round(elapsed, 4),
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "source_frame_id": frame_id,
            "frame_file": frame_name,
            "frame_policy": {"interval": interval, "scene_change": round(scene_change, 3)},
            "keys": keys,
            "player": {"minimap_position": [round(position[0], 6), round(position[1], 6)],
                       "found": player_found, "confidence": round(player_confidence, 4)},
            "capture": {"calibrated": calibrated, "minimap_bounds": bounds},
            "scene": scene,
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

    @staticmethod
    def _bounded(value: float) -> int:
        return int(round(max(0.0, min(100.0, value))))

    def _build_readiness_report(self, elapsed: float) -> Dict:
        samples = max(self._samples, 1)
        tracking = 100.0 * self._valid_player_samples / samples
        minimap = 100.0 * self._calibrated_samples / samples
        active = 100.0 * self._active_samples / samples
        frame_rate = self._frames_saved / max(elapsed, 1.0)
        movement_events = sum(self._key_down_counts[key] for key in self.MOVEMENT_KEYS)
        action_events = sum(self._key_down_counts[key] for key in self.ACTION_KEYS)
        knockbacks = self._event_counts["possible_knockback_or_forced_movement"]

        duration_score = self._bounded(elapsed / 900.0 * 100.0)
        tracking_score = self._bounded(tracking)
        minimap_score = self._bounded(minimap)
        activity_score = self._bounded(active * 3.0)
        movement_score = self._bounded(movement_events / 120.0 * 100.0)
        combat_score = self._bounded(action_events / 180.0 * 100.0)
        exception_score = self._bounded(knockbacks / 20.0 * 100.0)
        frame_score = self._bounded(frame_rate / 2.0 * 100.0)
        stability_penalty = min(30, self._black_frames * 3 + max(0, self._roi_changes - 3) * 2)

        quality_score = self._bounded(
            0.30 * tracking_score + 0.25 * minimap_score + 0.20 * frame_score
            + 0.15 * activity_score + 0.10 * duration_score - stability_penalty
        )
        navigation = self._bounded(0.35 * movement_score + 0.25 * tracking_score
                                   + 0.25 * minimap_score + 0.15 * duration_score)
        battle = self._bounded(0.50 * combat_score + 0.25 * activity_score
                               + 0.15 * duration_score + 0.10 * exception_score)
        exceptions = self._bounded(0.65 * exception_score + 0.35 * duration_score)

        model_ready = self._scene_model_ready_samples > 0
        vision = self._bounded(min(100.0, self._frames_saved / 1500.0 * 100.0))
        if not model_ready:
            # Raw frames are useful for detector training, but automatic monster
            # coverage cannot be claimed until classic_scene.pt exists.
            battle = min(battle, 55)

        overall = self._bounded(0.30 * vision + 0.30 * navigation + 0.30 * battle
                                + 0.10 * exceptions)
        stars = max(1, min(5, int(round(quality_score / 20.0))))
        suggestions = []
        if elapsed < 300:
            suggestions.append("再錄至少 5～15 分鐘，讓場景與操作更有多樣性。")
        if tracking < 95:
            suggestions.append("黃色菱形追蹤率不足 95%，請確認小地圖完整展開且定位穩定。")
        if movement_events < 80:
            suggestions.append("補錄左右移動、跳躍與上下梯，讓導航樣本更完整。")
        if action_events < 120:
            suggestions.append("補錄更多不同距離與方向的攻擊操作。")
        if knockbacks < 10:
            suggestions.append("可安全地補錄被怪物碰撞／擊退的情況，目標至少 10～20 次。")
        if not model_ready:
            suggestions.append("尚未有 classic_scene.pt；目前資料可用於人工標註與訓練怪物／梯子模型。")
        if self._black_frames:
            suggestions.append(f"偵測到 {self._black_frames} 張黑畫面，建議檢查遊戲最小化或切換視窗情況。")
        if not suggestions:
            suggestions.append("本 Session 品質良好，可保留並進入標註／訓練階段。")

        if quality_score < 45:
            recommendation = "建議刪除或重錄"
        elif overall < 55:
            recommendation = "保留；仍需補錄"
        elif overall < 75:
            recommendation = "可開始第一版訓練，建議再補錄"
        else:
            recommendation = "可開始訓練 AI"

        return {
            "format_version": 1,
            "quality_score": quality_score,
            "stars": stars,
            "overall_readiness": overall,
            "recommendation": recommendation,
            "coverage": {
                "visual_data": vision,
                "navigation_data": navigation,
                "battle_data": battle,
                "exception_events": exceptions,
            },
            "health": {
                "player_tracking_percent": round(tracking, 2),
                "minimap_tracking_percent": round(minimap, 2),
                "active_sample_percent": round(active, 2),
                "effective_frame_fps": round(frame_rate, 2),
                "black_frames": self._black_frames,
                "duplicate_frames_skipped": self._duplicate_skips,
                "minimap_roi_changes": self._roi_changes,
            },
            "behavior_counts": {
                "movement_key_down": movement_events,
                "action_key_down": action_events,
                "possible_knockback": knockbacks,
                "monster_detections": self._monster_samples,
                "ladder_detections": self._ladder_samples,
            },
            "scene_model_ready": model_ready,
            "duration_seconds": round(elapsed, 3),
            "samples": self._samples,
            "frames_saved": self._frames_saved,
            "size_mb": round(self._directory_size_mb(), 2),
            "suggestions": suggestions,
        }

    @staticmethod
    def _human_report(report: Dict) -> str:
        stars = "★" * report["stars"] + "☆" * (5 - report["stars"])
        coverage = report["coverage"]
        health = report["health"]
        lines = [
            "Auto Maple AI 資料就緒度報告",
            "=" * 34,
            f"資料品質：{stars}  {report['quality_score']}%",
            f"整體 AI 就緒度：{report['overall_readiness']}%",
            f"建議：{report['recommendation']}",
            "",
            f"視覺資料：{coverage['visual_data']}%",
            f"導航資料：{coverage['navigation_data']}%",
            f"戰鬥資料：{coverage['battle_data']}%",
            f"受擊／異常事件：{coverage['exception_events']}%",
            "",
            f"黃色菱形追蹤率：{health['player_tracking_percent']}%",
            f"小地圖追蹤率：{health['minimap_tracking_percent']}%",
            f"有效圖片：{report['frames_saved']} 張",
            f"資料容量：{report['size_mb']} MB",
            "",
            "建議補錄：",
        ]
        lines.extend(f"- {item}" for item in report["suggestions"])
        return "\n".join(lines) + "\n"

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
