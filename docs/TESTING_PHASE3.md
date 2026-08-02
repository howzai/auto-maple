# Phase 3 Windows validation checklist

1. Stand still and verify the reported position remains stable instead of flickering.
2. Walk normally left and right and verify coordinates move smoothly without large spikes.
3. Jump between platforms and verify normal jumps are accepted without excessive delay.
4. Use a portal or teleport and verify the new position is accepted after a brief confirmation period.
5. Place another similar minimap marker nearby and verify tracking prefers temporal continuity.
6. Hide or cover the player marker and verify automation pauses after the loss timeout.
7. Minimize or close the game and verify all held keys are released.
8. Move or resize the game window and verify minimap recalibration occurs.
9. Suspend the process or otherwise stall capture for more than 2.5 seconds and verify the watchdog pauses automation.
10. Inspect `health_snapshot()` and confirm frame age, confidence, candidate count, and error state are sensible.
