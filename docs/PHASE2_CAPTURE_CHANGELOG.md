# Phase 2: Capture and window safety

## Implemented

- The capture service now starts successfully even when MapleStory is not open.
- Missing, invalid, minimized, moved, resized, or closed game windows are handled safely.
- Automation is paused and all held keys are released when capture state becomes unsafe.
- Capture rate is bounded to 30 FPS instead of using a near-unbounded loop.
- Required image templates are loaded from a stable absolute project path and validated.
- Minimap calibration now checks normalized confidence and validates crop bounds.
- Player-marker loss is tracked and pauses automation after a short timeout.
- Capture publishes `window_found`, `player_found`, `frame_id`, and error state.
- Enabling a routine now requires a valid window, calibrated minimap, and visible player marker.
- Position recording is blocked when no valid player position is available.

## Manual validation required

1. Start without MapleStory open and confirm the GUI can still start.
2. Open MapleStory and verify minimap calibration succeeds.
3. Press Insert only after the player marker is visible.
4. Minimize or close MapleStory and confirm automation pauses immediately.
5. Cover or hide the player marker and confirm the bot pauses after the tracking timeout.
6. Move and resize the game window and confirm recalibration occurs.
7. Press F12 and verify every held movement key is released.

This phase intentionally avoids anti-cheat bypasses, memory access, injection, and packet manipulation.
