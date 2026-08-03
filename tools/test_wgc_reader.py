"""Display frames published by MapleCaptureHost.

Run this script from any working directory:
    python tools/test_wgc_reader.py

The project root is added to sys.path automatically so the src package can always
be imported. Press Esc or Q to close the preview window.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.modules.windows_graphics_capture import WindowsGraphicsCaptureReader

PREVIEW_MAX_WIDTH = 960
PREVIEW_MAX_HEIGHT = 540
WINDOW_NAME = "Auto Maple WGC transport test"


def resize_for_preview(frame):
    """Resize only the preview window while preserving the original frame data."""
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame

    scale = min(
        PREVIEW_MAX_WIDTH / width,
        PREVIEW_MAX_HEIGHT / height,
        1.0,
    )
    if scale >= 1.0:
        return frame

    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def main() -> int:
    reader = WindowsGraphicsCaptureReader()
    print("Waiting for MapleCaptureHost shared frames...")
    deadline = time.monotonic() + 15.0
    frames = 0
    started = time.monotonic()

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            frame = reader.read_latest()
            if frame is None:
                if time.monotonic() >= deadline and frames == 0:
                    print(reader.last_error or "No frame received within 15 seconds")
                    return 2
                time.sleep(0.005)
                continue

            frames += 1
            elapsed = max(time.monotonic() - started, 1e-6)
            fps = frames / elapsed

            preview = resize_for_preview(frame.copy())
            cv2.putText(
                preview,
                f"Shared memory FPS: {fps:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return 0
    finally:
        reader.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
