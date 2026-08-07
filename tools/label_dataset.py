"""Simple YOLO annotation tool for Auto Maple scene datasets.

Usage:
    python tools/label_dataset.py

Expected dataset layout (created by tools/build_dataset.py):
    training_dataset/images/{train,val,test}
    training_dataset/labels/{train,val,test}

Controls:
    1 player   2 monster   3 ladder   4 platform   5 obstacle
    Drag mouse to create a box using the selected class.
    Left/Right or A/D: previous/next image
    Delete/Backspace: delete selected box
    Ctrl+S: save current labels
    Space: save and go to next image

The tool remembers the last image for each split and autosaves after box edits, so
long manual-labeling sessions can be safely stopped and resumed.
"""

from __future__ import annotations

import argparse
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox
from typing import List, Optional, Tuple

from PIL import Image, ImageTk

CLASSES = ["player", "monster", "ladder", "platform", "obstacle"]
CLASS_COLORS = ["#00e5ff", "#39ff14", "#ffd700", "#ff7f50", "#ff3b30"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self, width: int, height: int) -> Tuple[float, float, float, float]:
        left, right = sorted((self.x1, self.x2))
        top, bottom = sorted((self.y1, self.y2))
        cx = ((left + right) / 2.0) / width
        cy = ((top + bottom) / 2.0) / height
        bw = (right - left) / width
        bh = (bottom - top) / height
        return cx, cy, bw, bh


class LabelApp:
    def __init__(self, root: tk.Tk, dataset_root: Path, split: str):
        self.root = root
        self.dataset_root = dataset_root
        self.split = split
        self.images_dir = dataset_root / "images" / split
        self.labels_dir = dataset_root / "labels" / split
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = dataset_root / f".labeler_progress_{split}.txt"

        self.images = sorted(
            path for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        self.index = self._load_progress()
        self.class_id = 1  # Monster-first workflow.
        self.boxes: List[Box] = []
        self.selected_index: Optional[int] = None
        self.start_point: Optional[Tuple[float, float]] = None
        self.preview_rect = None
        self.photo = None
        self.image = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        root.title("Auto Maple Dataset Labeler")
        root.geometry("1280x850")

        toolbar = tk.Frame(root)
        toolbar.pack(fill=tk.X, padx=8, pady=6)
        self.status = tk.StringVar()
        tk.Label(toolbar, textvariable=self.status, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.class_label = tk.StringVar()
        tk.Label(toolbar, textvariable=self.class_label, font=("Segoe UI", 11, "bold")).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(root, bg="#202020", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_select)
        self.canvas.bind("<Configure>", lambda _event: self.render())

        root.bind("1", lambda _e: self.set_class(0))
        root.bind("2", lambda _e: self.set_class(1))
        root.bind("3", lambda _e: self.set_class(2))
        root.bind("4", lambda _e: self.set_class(3))
        root.bind("5", lambda _e: self.set_class(4))
        root.bind("<Left>", lambda _e: self.previous())
        root.bind("a", lambda _e: self.previous())
        root.bind("<Right>", lambda _e: self.next())
        root.bind("d", lambda _e: self.next())
        root.bind("<space>", lambda _e: self.next(save=True))
        root.bind("<Delete>", lambda _e: self.delete_selected())
        root.bind("<BackSpace>", lambda _e: self.delete_selected())
        root.bind("<Control-s>", lambda _e: self.save_labels())
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.set_class(self.class_id)
        self.load_current()

    def _load_progress(self) -> int:
        try:
            value = int(self.progress_path.read_text(encoding="utf-8").strip())
        except Exception:
            return 0
        return min(max(0, value), max(0, len(self.images) - 1))

    def _save_progress(self):
        try:
            self.progress_path.write_text(str(self.index), encoding="utf-8")
        except OSError:
            pass

    def set_class(self, class_id: int):
        self.class_id = class_id
        self.class_label.set(f"Class {class_id}: {CLASSES[class_id]}")

    def current_image_path(self) -> Path:
        return self.images[self.index]

    def current_label_path(self) -> Path:
        return self.labels_dir / f"{self.current_image_path().stem}.txt"

    def load_current(self):
        self._save_progress()
        self.image = Image.open(self.current_image_path()).convert("RGB")
        self.boxes = []
        label_path = self.current_label_path()
        if label_path.exists():
            width, height = self.image.size
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                if 0 <= class_id < len(CLASSES):
                    self.boxes.append(Box(class_id, x1, y1, x2, y2))
        self.selected_index = None
        self.update_status()
        self.render()

    def update_status(self):
        reviewed = sum(1 for image in self.images if (self.labels_dir / f"{image.stem}.txt").exists())
        monster_boxes = sum(1 for box in self.boxes if box.class_id == 1)
        self.status.set(
            f"{self.split}  {self.index + 1}/{len(self.images)}  "
            f"Reviewed files: {reviewed}  {self.current_image_path().name}  "
            f"Boxes: {len(self.boxes)} (monster {monster_boxes})  "
            "[1-5 class | drag add | right-click select | Del remove | Space save+next]"
        )

    def render(self):
        if self.image is None:
            return
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        iw, ih = self.image.size
        self.scale = min(cw / iw, ch / ih)
        display_size = (max(1, int(iw * self.scale)), max(1, int(ih * self.scale)))
        display = self.image.resize(display_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        self.offset_x = (cw - display_size[0]) // 2
        self.offset_y = (ch - display_size[1]) // 2
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor=tk.NW)

        for i, box in enumerate(self.boxes):
            x1, y1 = self.to_canvas(box.x1, box.y1)
            x2, y2 = self.to_canvas(box.x2, box.y2)
            color = CLASS_COLORS[box.class_id]
            width = 4 if i == self.selected_index else 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
            self.canvas.create_text(
                min(x1, x2) + 4, min(y1, y2) + 4,
                text=CLASSES[box.class_id], fill=color, anchor=tk.NW,
                font=("Segoe UI", 10, "bold")
            )

    def to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return self.offset_x + x * self.scale, self.offset_y + y * self.scale

    def to_image(self, x: float, y: float) -> Tuple[float, float]:
        iw, ih = self.image.size
        ix = min(max((x - self.offset_x) / self.scale, 0), iw)
        iy = min(max((y - self.offset_y) / self.scale, 0), ih)
        return ix, iy

    def on_press(self, event):
        self.start_point = self.to_image(event.x, event.y)
        self.preview_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=CLASS_COLORS[self.class_id], width=2, dash=(5, 3)
        )

    def on_drag(self, event):
        if self.preview_rect is not None:
            x0, y0 = self.to_canvas(*self.start_point)
            self.canvas.coords(self.preview_rect, x0, y0, event.x, event.y)

    def on_release(self, event):
        if self.start_point is None:
            return
        end = self.to_image(event.x, event.y)
        x1, y1 = self.start_point
        x2, y2 = end
        self.start_point = None
        self.preview_rect = None
        if abs(x2 - x1) >= 4 and abs(y2 - y1) >= 4:
            self.boxes.append(Box(self.class_id, x1, y1, x2, y2))
            self.selected_index = len(self.boxes) - 1
            self.save_labels()  # Autosave immediately after every new box.
        self.update_status()
        self.render()

    def on_select(self, event):
        px, py = self.to_image(event.x, event.y)
        self.selected_index = None
        for i in range(len(self.boxes) - 1, -1, -1):
            box = self.boxes[i]
            left, right = sorted((box.x1, box.x2))
            top, bottom = sorted((box.y1, box.y2))
            if left <= px <= right and top <= py <= bottom:
                self.selected_index = i
                self.set_class(box.class_id)
                break
        self.render()

    def delete_selected(self):
        if self.selected_index is not None and 0 <= self.selected_index < len(self.boxes):
            del self.boxes[self.selected_index]
            self.selected_index = None
            self.save_labels()
            self.update_status()
            self.render()

    def save_labels(self):
        if self.image is None:
            return
        width, height = self.image.size
        lines = []
        for box in self.boxes:
            cx, cy, bw, bh = box.normalized(width, height)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{box.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        self.current_label_path().write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self._save_progress()
        self.update_status()

    def previous(self):
        self.save_labels()
        self.index = max(0, self.index - 1)
        self.load_current()

    def next(self, save: bool = True):
        if save:
            self.save_labels()
        if self.index < len(self.images) - 1:
            self.index += 1
            self.load_current()
        else:
            messagebox.showinfo("Auto Maple Dataset Labeler", "This is the last image in this split.")

    def on_close(self):
        self.save_labels()
        self._save_progress()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Label Auto Maple YOLO scene images")
    parser.add_argument("--dataset", default="training_dataset", help="Dataset root")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    images_dir = dataset_root / "images" / args.split
    if not images_dir.is_dir():
        raise SystemExit(
            f"Dataset images not found: {images_dir}\n"
            "Run build_dataset.bat first."
        )

    root = tk.Tk()
    try:
        LabelApp(root, dataset_root, args.split)
        root.mainloop()
    except Exception as exc:
        root.destroy()
        raise SystemExit(f"Label tool failed: {exc}") from exc


if __name__ == "__main__":
    main()
