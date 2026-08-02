# Phase 1 Changelog

## Runtime startup

- Added bounded startup waits for Bot, Capture, Notifier, and Listener.
- Startup now reports dead initialization threads and timeout failures.
- Shutdown always disables automation and releases tracked keys.

## Input safety

- Added tracking for keys pressed by the application.
- Added `release_all()` for pause, failure, emergency stop, and process exit.
- Added F12 emergency stop.
- Added edge-triggered hotkeys to prevent repeated toggles while a key is held.

## Listener reliability

- Added exception containment around the keyboard listener loop.
- Added minimap recalibration timeout.
- Pausing now releases all held keys.
