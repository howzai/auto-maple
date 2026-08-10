"""Live read-only test for the trained monster and navigation YOLO models.

This tool does NOT press keys or control the game. It captures the visible
MapleStory client area, runs both trained models on the same BGR frame (matching
OpenCV/Ultralytics numpy conventions), and overlays detections for validation.

Models:
    assets/models/classic_scene.pt      -> monster
    assets/models/navigation_scene.pt   -> ladder / platform

Press F10 or Escape in the preview window to close.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = Path(__file__).resolve().parents[1]
MONSTER_MODEL = ROOT / "assets" / "models" / "classic_scene.pt"
NAV_MODEL = ROOT / "assets" / "models" / "navigation_scene.pt"


def _find_maple_window():
    try:
        import win32gui
    except Exception:
        return None
    candidates = []

    def callback(hwnd, _extra):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = (win32gui.GetWindowText(hwnd) or "").strip()
        folded = title.casefold()
        if title and ("楓之谷" in title or "maplestory" in folded or "新楓" in title):
            candidates.append((hwnd, title))

    win32gui.EnumWindows(callback, None)
    best = None
    for hwnd, title in candidates:
        try:
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            area = max(0, right - left) * max(0, bottom - top)
        except Exception:
            continue
        if best is None or area > best[0]:
            best = (area, hwnd, title)
    return None if best is None else (best[1], best[2])


def _client_rect_on_screen(hwnd):
    import win32gui
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    sx1, sy1 = win32gui.ClientToScreen(hwnd, (left, top))
    sx2, sy2 = win32gui.ClientToScreen(hwnd, (right, bottom))
    width, height = sx2 - sx1, sy2 - sy1
    if width <= 64 or height <= 64:
        return None
    return {"left": sx1, "top": sy1, "width": width, "height": height}


def _font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_box(draw, xyxy, text, outline, width=3):
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    draw.rectangle((x1, y1, x2, y2), outline=outline, width=width)
    font = _font(18)
    try:
        bbox = draw.textbbox((x1, y1), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = 100, 18
    ty = max(0, y1 - th - 4)
    draw.rectangle((x1, ty, x1 + tw + 6, ty + th + 4), fill=(0, 0, 0))
    draw.text((x1 + 3, ty + 2), text, fill=outline, font=font)


class CombinedVisionApp:
    def __init__(self, monster_conf, nav_conf, imgsz):
        try:
            import torch
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(f"YOLO/PyTorch unavailable: {exc}") from exc

        if not MONSTER_MODEL.is_file():
            raise RuntimeError(f"Monster model not found: {MONSTER_MODEL}")
        if not NAV_MODEL.is_file():
            raise RuntimeError(f"Navigation model not found: {NAV_MODEL}")

        found = _find_maple_window()
        if found is None:
            raise RuntimeError("MapleStory window was not found. Open the game first.")
        self.hwnd, self.game_title = found
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.monster = YOLO(str(MONSTER_MODEL))
        self.navigation = YOLO(str(NAV_MODEL))
        self.monster_conf = monster_conf
        self.nav_conf = nav_conf
        self.imgsz = imgsz

        import mss
        self.sct = mss.mss()
        self.root = tk.Tk()
        self.root.title("Auto Maple AI - Combined Vision Test (F10 / Esc to close)")
        self.root.configure(bg="black")
        self.root.bind("<F10>", lambda _e: self.close())
        self.root.bind("<Escape>", lambda _e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.StringVar(value="Loading models...")
        tk.Label(self.root, textvariable=self.status, bg="black", fg="white",
                 anchor="w", font=("Segoe UI", 11, "bold")).pack(fill=tk.X, padx=6, pady=4)
        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.photo = None
        self.closed = False
        self.last_error = ""
        self.root.after(10, self.tick)

    def close(self):
        self.closed = True
        try:
            self.sct.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def tick(self):
        if self.closed:
            return
        started = time.perf_counter()
        try:
            rect = _client_rect_on_screen(self.hwnd)
        except Exception:
            rect = None
        if rect is None:
            self.status.set("Waiting for visible game window...")
            self.root.after(250, self.tick)
            return

        try:
            shot = self.sct.grab(rect)
            # MSS returns BGRA. Keep BGR for YOLO numpy inference so this matches
            # the OpenCV-style images used by the existing runtime and training.
            frame_bgr = np.asarray(shot, dtype=np.uint8)[:, :, :3].copy()

            nav_start = time.perf_counter()
            nav_results = self.navigation.predict(
                frame_bgr, imgsz=self.imgsz, conf=self.nav_conf,
                device=self.device, verbose=False,
            )
            nav_ms = (time.perf_counter() - nav_start) * 1000.0

            monster_start = time.perf_counter()
            monster_results = self.monster.predict(
                frame_bgr, imgsz=self.imgsz, conf=self.monster_conf,
                device=self.device, verbose=False,
            )
            monster_ms = (time.perf_counter() - monster_start) * 1000.0

            # PIL expects RGB, so convert only for the preview display.
            image = Image.fromarray(frame_bgr[:, :, ::-1])
            draw = ImageDraw.Draw(image)
            monsters = ladders = platforms = 0

            for result in monster_results:
                boxes = getattr(result, "boxes", None)
                names = getattr(result, "names", {})
                if boxes is None:
                    continue
                for box in boxes:
                    class_id = int(box.cls[0])
                    label = str(names.get(class_id, class_id)).casefold().strip()
                    if label not in {"monster", "mob", "enemy", "0"} and len(names) > 1:
                        continue
                    conf = float(box.conf[0])
                    _draw_box(draw, box.xyxy[0].tolist(), f"monster {conf:.0%}", (57, 255, 20))
                    monsters += 1

            for result in nav_results:
                boxes = getattr(result, "boxes", None)
                names = getattr(result, "names", {})
                if boxes is None:
                    continue
                for box in boxes:
                    class_id = int(box.cls[0])
                    label = str(names.get(class_id, class_id)).casefold().strip()
                    conf = float(box.conf[0])
                    if label == "ladder" or (len(names) == 2 and class_id == 0):
                        _draw_box(draw, box.xyxy[0].tolist(), f"ladder {conf:.0%}", (255, 215, 0))
                        ladders += 1
                    elif label == "platform" or (len(names) == 2 and class_id == 1):
                        _draw_box(draw, box.xyxy[0].tolist(), f"platform {conf:.0%}", (255, 127, 80))
                        platforms += 1

            total_ms = (time.perf_counter() - started) * 1000.0
            self.status.set(
                f"GPU: {self.device} | Monsters: {monsters} | Ladders: {ladders} | "
                f"Platforms: {platforms} | Monster: {monster_ms:.1f} ms | "
                f"Nav: {nav_ms:.1f} ms | Total: {total_ms:.1f} ms"
            )

            max_w, max_h = 1500, 850
            scale = min(max_w / image.width, max_h / image.height, 1.0)
            if scale < 1.0:
                image = image.resize((max(1, int(image.width * scale)),
                                      max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)
            self.last_error = ""
        except Exception as exc:
            text = f"Combined vision error: {exc}"
            if text != self.last_error:
                print(f"[ERROR] {text}")
                self.last_error = text
            self.status.set(text)
        self.root.after(1, self.tick)

    def run(self):
        print("\n========================================")
        print("  Auto Maple Combined Vision Test")
        print("========================================")
        print(f"Game: {self.game_title}")
        print(f"Monster model: {MONSTER_MODEL}")
        print(f"Navigation model: {NAV_MODEL}")
        print(f"Device: {self.device}")
        print(f"Monster confidence: {self.monster_conf:.2f}")
        print(f"Navigation confidence: {self.nav_conf:.2f}")
        print("Inference color mode: BGR (OpenCV/Ultralytics numpy)")
        print("This mode is READ ONLY. It never presses game keys.")
        print("Press F10 or Escape in the preview window to close.\n")
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Live combined Auto Maple YOLO test")
    parser.add_argument("--monster-conf", type=float, default=0.45)
    parser.add_argument("--nav-conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    try:
        app = CombinedVisionApp(
            monster_conf=min(max(args.monster_conf, 0.05), 0.99),
            nav_conf=min(max(args.nav_conf, 0.05), 0.99),
            imgsz=max(320, args.imgsz),
        )
        app.run()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        try:
            root = tk.Tk(); root.withdraw()
            messagebox.showerror("Auto Maple Combined Vision", str(exc))
            root.destroy()
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
