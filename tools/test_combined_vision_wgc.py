"""Low-latency read-only combined vision test using Windows Graphics Capture.

Production-like scheduler:
  * monster model:     20 Hz
  * navigation model:  4 Hz
  * debug preview:    10 Hz

The latest detections are cached between inference passes. The preview uses OpenCV
rather than Tk/PIL conversion, so debug rendering cannot dominate the test loop.
No game input is sent.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.windows_graphics_capture import WindowsGraphicsCaptureReader
from src.modules.wgc_capture_backend import CaptureHostController

MONSTER_MODEL = ROOT / "assets" / "models" / "classic_scene.pt"
NAV_MODEL = ROOT / "assets" / "models" / "navigation_scene.pt"
WINDOW = "Auto Maple AI - Combined WGC Vision Test"
VK_F10 = 0x79


def _draw_box(frame: np.ndarray, xyxy, text: str, color) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        text,
        (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        1,
        cv2.LINE_AA,
    )


def _extract_monsters(results):
    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            label = str(names.get(class_id, class_id)).casefold().strip()
            if label not in {"monster", "mob", "enemy", "0"} and len(names) > 1:
                continue
            detections.append((box.xyxy[0].tolist(), float(box.conf[0])))
    return detections


def _extract_navigation(results):
    ladders = []
    platforms = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            label = str(names.get(class_id, class_id)).casefold().strip()
            item = (box.xyxy[0].tolist(), float(box.conf[0]))
            if label == "ladder" or (len(names) == 2 and class_id == 0):
                ladders.append(item)
            elif label == "platform" or (len(names) == 2 and class_id == 1):
                platforms.append(item)
    return ladders, platforms


def _f10_pressed() -> bool:
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(VK_F10) & 0x8000)
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Production-like combined WGC vision test")
    parser.add_argument("--monster-conf", type=float, default=0.45)
    parser.add_argument("--nav-conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--monster-hz", type=float, default=20.0)
    parser.add_argument("--nav-hz", type=float, default=4.0)
    parser.add_argument("--preview-hz", type=float, default=10.0)
    args = parser.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        print(f"[ERROR] YOLO/PyTorch unavailable: {exc}")
        return 1

    if not MONSTER_MODEL.is_file():
        print(f"[ERROR] Monster model not found: {MONSTER_MODEL}")
        return 1
    if not NAV_MODEL.is_file():
        print(f"[ERROR] Navigation model not found: {NAV_MODEL}")
        return 1

    device = "0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    monster = YOLO(str(MONSTER_MODEL))
    navigation = YOLO(str(NAV_MODEL))

    monster_interval = 1.0 / max(1.0, args.monster_hz)
    nav_interval = 1.0 / max(0.5, args.nav_hz)
    preview_interval = 1.0 / max(1.0, args.preview_hz)
    monster_conf = min(max(args.monster_conf, 0.05), 0.99)
    nav_conf = min(max(args.nav_conf, 0.05), 0.99)
    imgsz = max(320, args.imgsz)

    host = CaptureHostController()
    reader = WindowsGraphicsCaptureReader()

    monsters = []
    ladders = []
    platforms = []
    monster_ms = 0.0
    nav_ms = 0.0
    last_monster = 0.0
    last_nav = 0.0
    last_preview = 0.0
    preview_fps = 0.0
    last_preview_stamp = 0.0
    latest_bgr = None

    print("\n========================================")
    print("  Auto Maple Combined WGC Vision Test")
    print("========================================")
    print(f"GPU: {gpu_name} (device {device})")
    print(f"Monster: {args.monster_hz:.1f} Hz | Navigation: {args.nav_hz:.1f} Hz | Preview: {args.preview_hz:.1f} Hz")
    print("Capture: Windows Graphics Capture shared memory")
    print("READ ONLY: no game keys are sent.")
    print("Press F10, Esc, or Q to close.\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    try:
        while True:
            if _f10_pressed():
                break

            if not host.ensure_running():
                time.sleep(0.05)
                continue

            frame = reader.read_latest()
            if frame is not None:
                latest_bgr = frame[:, :, :3].copy()

            if latest_bgr is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                time.sleep(0.002)
                continue

            now = time.perf_counter()

            if now - last_monster >= monster_interval:
                started = time.perf_counter()
                results = monster.predict(
                    latest_bgr,
                    imgsz=imgsz,
                    conf=monster_conf,
                    device=device,
                    verbose=False,
                )
                monster_ms = (time.perf_counter() - started) * 1000.0
                monsters = _extract_monsters(results)
                last_monster = now

            if now - last_nav >= nav_interval:
                started = time.perf_counter()
                results = navigation.predict(
                    latest_bgr,
                    imgsz=imgsz,
                    conf=nav_conf,
                    device=device,
                    verbose=False,
                )
                nav_ms = (time.perf_counter() - started) * 1000.0
                ladders, platforms = _extract_navigation(results)
                last_nav = now

            if now - last_preview >= preview_interval:
                preview = latest_bgr.copy()
                for xyxy, conf in monsters:
                    _draw_box(preview, xyxy, f"monster {conf:.0%}", (20, 255, 57))
                for xyxy, conf in ladders:
                    _draw_box(preview, xyxy, f"ladder {conf:.0%}", (0, 215, 255))
                for xyxy, conf in platforms:
                    _draw_box(preview, xyxy, f"platform {conf:.0%}", (80, 127, 255))

                h, w = preview.shape[:2]
                cv2.drawMarker(preview, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

                if last_preview_stamp:
                    instant = 1.0 / max(now - last_preview_stamp, 1e-6)
                    preview_fps = instant if preview_fps <= 0 else preview_fps * 0.8 + instant * 0.2
                last_preview_stamp = now

                hud1 = (
                    f"GPU: {gpu_name} | Preview: {preview_fps:.1f} FPS | "
                    f"Monsters: {len(monsters)} | Ladders: {len(ladders)} | Platforms: {len(platforms)}"
                )
                hud2 = (
                    f"Monster: {monster_ms:.1f} ms @ {args.monster_hz:.0f}Hz | "
                    f"Nav: {nav_ms:.1f} ms @ {args.nav_hz:.0f}Hz"
                )
                cv2.rectangle(preview, (8, 8), (min(w - 8, 1040), 66), (0, 0, 0), -1)
                cv2.putText(preview, hud1, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(preview, hud2, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

                target_w = min(1280, w)
                target_h = max(360, int(h * target_w / max(1, w)))
                cv2.resizeWindow(WINDOW, target_w, target_h)
                cv2.imshow(WINDOW, preview)
                last_preview = now

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        host.stop()
        try:
            cv2.destroyWindow(WINDOW)
            cv2.waitKey(1)
        except cv2.error:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
