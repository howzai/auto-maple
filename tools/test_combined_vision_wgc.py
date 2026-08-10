"""Low-latency read-only combined vision test using Windows Graphics Capture.

This tester reads MapleStory frames directly from MapleCaptureHost shared memory,
so its own preview window is never captured recursively.  Monster inference runs
more often than navigation inference because ladders/platforms are effectively
static scene geometry.

Models:
    assets/models/classic_scene.pt      -> monster
    assets/models/navigation_scene.pt   -> ladder / platform

No game keys are ever sent. Press F10 or Escape to close.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.modules.windows_graphics_capture import WindowsGraphicsCaptureReader
from src.modules.wgc_capture_backend import CaptureHostController

ROOT = Path(__file__).resolve().parents[1]
MONSTER_MODEL = ROOT / "assets" / "models" / "classic_scene.pt"
NAV_MODEL = ROOT / "assets" / "models" / "navigation_scene.pt"


def _draw_box(frame: np.ndarray, xyxy, text: str, color) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    ty = max(th + 4, y1)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x1 + 2, ty - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
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


class CombinedWgcVisionApp:
    def __init__(
        self,
        monster_conf: float,
        nav_conf: float,
        imgsz: int,
        monster_every: int,
        nav_every: int,
    ):
        try:
            import torch
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(f"YOLO/PyTorch unavailable: {exc}") from exc

        if not MONSTER_MODEL.is_file():
            raise RuntimeError(f"Monster model not found: {MONSTER_MODEL}")
        if not NAV_MODEL.is_file():
            raise RuntimeError(f"Navigation model not found: {NAV_MODEL}")

        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.gpu_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )
        self.monster = YOLO(str(MONSTER_MODEL))
        self.navigation = YOLO(str(NAV_MODEL))
        self.monster_conf = monster_conf
        self.nav_conf = nav_conf
        self.imgsz = imgsz
        self.monster_every = max(1, monster_every)
        self.nav_every = max(1, nav_every)

        self.host = CaptureHostController()
        self.reader = WindowsGraphicsCaptureReader()

        self.monsters = []
        self.ladders = []
        self.platforms = []
        self.monster_ms = 0.0
        self.nav_ms = 0.0
        self.frame_index = 0
        self.last_frame_time = 0.0
        self.display_fps = 0.0
        self.closed = False
        self.photo = None
        self.last_error = ""

        self.root = tk.Tk()
        self.root.title("Auto Maple AI - Combined WGC Vision Test (F10 / Esc to close)")
        self.root.configure(bg="black")
        self.root.bind("<F10>", lambda _e: self.close())
        self.root.bind("<Escape>", lambda _e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status = tk.StringVar(value="Starting Windows Graphics Capture...")
        tk.Label(
            self.root,
            textvariable=self.status,
            bg="black",
            fg="white",
            anchor="w",
            font=("Segoe UI", 11, "bold"),
        ).pack(fill=tk.X, padx=6, pady=4)
        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.root.after(10, self.tick)

    def close(self):
        self.closed = True
        try:
            self.reader.close()
        except Exception:
            pass
        try:
            self.host.stop()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def _next_frame(self):
        if not self.host.ensure_running():
            return None
        return self.reader.read_latest()

    def tick(self):
        if self.closed:
            return

        frame_started = time.perf_counter()
        frame = self._next_frame()
        if frame is None:
            detail = self.reader.last_error or self.host.last_error or "Waiting for WGC frame"
            self.status.set(f"WGC waiting: {detail}")
            self.root.after(2, self.tick)
            return

        try:
            # WGC publishes BGRA. Ultralytics/OpenCV numpy inputs use BGR.
            bgr = frame[:, :, :3].copy()

            if self.frame_index % self.monster_every == 0:
                started = time.perf_counter()
                results = self.monster.predict(
                    bgr,
                    imgsz=self.imgsz,
                    conf=self.monster_conf,
                    device=self.device,
                    verbose=False,
                )
                self.monster_ms = (time.perf_counter() - started) * 1000.0
                self.monsters = _extract_monsters(results)

            if self.frame_index % self.nav_every == 0:
                started = time.perf_counter()
                results = self.navigation.predict(
                    bgr,
                    imgsz=self.imgsz,
                    conf=self.nav_conf,
                    device=self.device,
                    verbose=False,
                )
                self.nav_ms = (time.perf_counter() - started) * 1000.0
                self.ladders, self.platforms = _extract_navigation(results)

            preview = bgr.copy()
            for xyxy, conf in self.monsters:
                _draw_box(preview, xyxy, f"monster {conf:.0%}", (20, 255, 57))
            for xyxy, conf in self.ladders:
                _draw_box(preview, xyxy, f"ladder {conf:.0%}", (0, 215, 255))
            for xyxy, conf in self.platforms:
                _draw_box(preview, xyxy, f"platform {conf:.0%}", (80, 127, 255))

            now = time.perf_counter()
            if self.last_frame_time:
                instant = 1.0 / max(now - self.last_frame_time, 1e-6)
                self.display_fps = instant if self.display_fps <= 0 else self.display_fps * 0.85 + instant * 0.15
            self.last_frame_time = now

            total_ms = (now - frame_started) * 1000.0
            self.status.set(
                f"GPU: {self.gpu_name} | FPS: {self.display_fps:.1f} | "
                f"Monsters: {len(self.monsters)} | Ladders: {len(self.ladders)} | "
                f"Platforms: {len(self.platforms)} | Monster: {self.monster_ms:.1f} ms/{self.monster_every}f | "
                f"Nav: {self.nav_ms:.1f} ms/{self.nav_every}f | Frame: {total_ms:.1f} ms"
            )

            # Fast debug preview. Inference always uses the original full-size frame.
            max_w, max_h = 1280, 720
            h, w = preview.shape[:2]
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                preview = cv2.resize(
                    preview,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.image_label.configure(image=self.photo)
            self.last_error = ""
        except Exception as exc:
            text = f"Combined WGC vision error: {exc}"
            if text != self.last_error:
                print(f"[ERROR] {text}")
                self.last_error = text
            self.status.set(text)

        self.frame_index += 1
        self.root.after(1, self.tick)

    def run(self):
        print("\n========================================")
        print("  Auto Maple Combined WGC Vision Test")
        print("========================================")
        print(f"Monster model: {MONSTER_MODEL}")
        print(f"Navigation model: {NAV_MODEL}")
        print(f"GPU: {self.gpu_name} (device {self.device})")
        print(f"Monster inference cadence: every {self.monster_every} frame(s)")
        print(f"Navigation inference cadence: every {self.nav_every} frame(s)")
        print("Capture: Windows Graphics Capture shared memory (no desktop/MSS recursion)")
        print("READ ONLY: no game keys are sent.")
        print("Press F10 or Escape in the preview window to close.\n")
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Low-latency combined Auto Maple WGC test")
    parser.add_argument("--monster-conf", type=float, default=0.45)
    parser.add_argument("--nav-conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--monster-every", type=int, default=2)
    parser.add_argument("--nav-every", type=int, default=5)
    args = parser.parse_args()

    try:
        app = CombinedWgcVisionApp(
            monster_conf=min(max(args.monster_conf, 0.05), 0.99),
            nav_conf=min(max(args.nav_conf, 0.05), 0.99),
            imgsz=max(320, args.imgsz),
            monster_every=max(1, args.monster_every),
            nav_every=max(1, args.nav_every),
        )
        app.run()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Auto Maple Combined WGC Vision", str(exc))
            root.destroy()
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
