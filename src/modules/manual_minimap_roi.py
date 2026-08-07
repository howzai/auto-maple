"""Persistent user-defined minimap search region for the classic client.

The saved rectangle is relative to the WGC game frame, not the desktop.  It is
intentionally a SEARCH REGION (title bar + minimap + a little margin), not the
final minimap canvas.  Map-specific canvas geometry is re-detected only inside
this small region, making startup and map changes much faster and safer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


Bounds = Tuple[Tuple[int, int], Tuple[int, int]]
CONFIG_PATH = Path("settings") / "classic_minimap_roi.json"


def _valid(frame: np.ndarray, bounds: Bounds) -> bool:
    (x1, y1), (x2, y2) = bounds
    h, w = frame.shape[:2]
    return 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and (x2 - x1) >= 120 and (y2 - y1) >= 90


def load_search_region(frame: np.ndarray) -> Optional[Bounds]:
    if frame is None or frame.size == 0 or not CONFIG_PATH.is_file():
        return None
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        rect = data.get("search_region", {})
        # Store normalized coordinates as well as source dimensions so the same
        # calibration survives moving the window and modest resolution changes.
        if all(k in rect for k in ("nx1", "ny1", "nx2", "ny2")):
            h, w = frame.shape[:2]
            bounds = (
                (int(round(float(rect["nx1"]) * w)), int(round(float(rect["ny1"]) * h))),
                (int(round(float(rect["nx2"]) * w)), int(round(float(rect["ny2"]) * h))),
            )
        else:
            bounds = (
                (int(rect["x1"]), int(rect["y1"])),
                (int(rect["x2"]), int(rect["y2"])),
            )
        return bounds if _valid(frame, bounds) else None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def save_search_region(frame: np.ndarray, bounds: Bounds) -> None:
    if not _valid(frame, bounds):
        raise ValueError("Invalid minimap search region")
    (x1, y1), (x2, y2) = bounds
    h, w = frame.shape[:2]
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "frame_width": w,
        "frame_height": h,
        "search_region": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "nx1": x1 / float(w),
            "ny1": y1 / float(h),
            "nx2": x2 / float(w),
            "ny2": y2 / float(h),
        },
    }
    CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_search_region(capture) -> bool:
    """Let the user drag a rectangle over the minimap area on a WGC frame."""
    frame = capture.screenshot(delay=0)
    if frame is None or frame.size == 0:
        print("\n[!] F11 calibration failed: no WGC game frame is available")
        return False

    display = frame[:, :, :3].copy()
    print("\n[~] F11 minimap calibration")
    print("[~] Drag a box around the ENTIRE minimap panel (title + map + a little margin).")
    print("[~] Press ENTER/SPACE to save, or ESC to cancel.")
    try:
        x, y, w, h = cv2.selectROI("Auto Maple - Select Minimap Area", display, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Auto Maple - Select Minimap Area")
    except cv2.error as exc:
        print(f"\n[!] Minimap calibration window failed: {exc}")
        return False

    if w <= 0 or h <= 0:
        print("\n[~] Minimap calibration cancelled")
        return False

    bounds = ((int(x), int(y)), (int(x + w), int(y + h)))
    if not _valid(frame, bounds):
        print("\n[!] Selected area is too small or outside the game frame")
        return False

    save_search_region(frame, bounds)
    capture.calibrated = False
    (x1, y1), (x2, y2) = bounds
    print(f"\n[~] Saved minimap search region: ({x1},{y1})-({x2},{y2})")
    print("[~] Auto Maple will now search only inside this region; map changes should recover much faster.")
    return True
