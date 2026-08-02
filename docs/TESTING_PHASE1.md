# Phase 1 Manual Test Checklist

Run these checks on Windows after installing the project dependencies.

1. Start the application normally and confirm all modules initialize.
2. Start without MapleStory open and confirm startup fails clearly instead of waiting forever.
3. Enable the routine, hold a movement action, then press F12.
4. Confirm automation disables immediately and the character stops moving.
5. Pause with Insert and confirm no key remains held.
6. Trigger minimap recalibration with the game unavailable and confirm it times out.
7. Close the program with Ctrl+C and confirm all keys are released.

Record any traceback and the exact Windows, Python, TensorFlow, NumPy, OpenCV, and MSS versions when reporting a failure.
