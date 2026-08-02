# Phase 2 known risks

- Exact window title matching may need configuration for regional game clients.
- Template confidence thresholds may require tuning after game UI changes.
- Windows display scaling and full-screen modes still require real-device validation.
- The original GUI may assume that minimap data is always populated; startup without the game must be tested.
- The original TensorFlow dependency remains unchanged in this phase.
