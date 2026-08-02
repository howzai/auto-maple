# Phase 5: GUI Health Panel

The View tab now displays live runtime diagnostics:

- runtime/capture state
- enabled or paused mode
- measured capture FPS
- player marker confidence
- normalized player position
- age of the most recent frame
- last capture error

Tkinter refreshes were moved from a background worker to `root.after(...)` callbacks so all widget access occurs on the Tk main thread. This avoids intermittent Tcl/Tk crashes and race conditions.

## Windows checks

1. Start without MapleStory and confirm `Waiting for game window`.
2. Open the game and confirm calibration/player states change automatically.
3. Verify FPS, confidence, position, and frame age update without GUI freezing.
4. Minimize or close the game and confirm the panel reports the changed state.
5. Trigger F12 and confirm the runtime mode changes to `Paused`.
6. Close the GUI and confirm capture is stopped and held keys are released.
