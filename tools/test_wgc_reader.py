"""Display frames published by MapleCaptureHost --test-pattern.

Run the host first:
    dotnet run --project capture_host -- --test-pattern
Then run this script from the project root:
    python tools/test_wgc_reader.py
Press Esc to close.
"""

from __future__ import annotations

import time

import cv2

from src.modules.windows_graphics_capture import WindowsGraphicsCaptureReader


def main() -> int:
    reader = WindowsGraphicsCaptureReader()
    print("Waiting for MapleCaptureHost shared frames...")
    deadline = time.monotonic() + 15.0
    frames = 0
    started = time.monotonic()

    try:
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
            preview = frame.copy()
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
            cv2.imshow("Auto Maple WGC transport test", preview)
            if cv2.waitKey(1) & 0xFF == 27:
                return 0
    finally:
        reader.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
