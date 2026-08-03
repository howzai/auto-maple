# MapleCaptureHost

This helper will provide a dedicated Windows Graphics Capture backend for Auto Maple.

## Goal

- Bind to the `新楓之谷：經典版` window by HWND.
- Capture only that target window.
- Never silently fall back to desktop capture in background mode.
- Publish the newest BGRA frame to Python through a bounded shared-memory buffer.
- Include width, height, stride, frame id, timestamp, and health status.
- Drop old frames instead of queuing them, preventing latency buildup.

## Current status

The project skeleton and target-window binding are implemented.

Next implementation steps:

1. Create `GraphicsCaptureItem` from HWND.
2. Create a D3D11 device and `Direct3D11CaptureFramePool`.
3. Copy the newest BGRA frame into shared memory.
4. Add the Python shared-memory reader.
5. Replace the background MSS/PrintWindow fallback with an explicit capture mode.
6. Add backend, FPS, frame age, and failure reason to the GUI.

## Planned safety behavior

When the selected window stops producing valid frames, Auto Maple will pause and release held keys. It will not substitute Chrome, CMD, taskbar previews, or other desktop pixels.
