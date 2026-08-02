# Auto Maple Optimization Roadmap

This fork is being modernized incrementally so each step stays reviewable and reversible.

## Phase 1 — Runtime safety

- Bounded component startup instead of infinite waits
- Emergency stop hotkey
- Guaranteed release of keys held by the application
- Input key debounce
- Safer minimap recalibration timeout
- Clear startup failure reporting

## Phase 2 — Capture and window safety

- Detect missing or closed game windows
- Pause when the target window is not usable
- Limit capture frame rate
- Validate image assets before use
- Publish immutable capture snapshots

## Phase 3 — Vision and tracking

- Minimap calibration confidence
- Player position smoothing
- Outlier rejection
- Recalibration after resolution or window changes
- Debug overlays and replay-based tests

## Phase 4 — Routine and command architecture

- Routine validation before execution
- Action timeouts and recovery states
- Explicit skill cooldown scheduling
- Structured logs and diagnostics

## Safety scope

The project will use ordinary screen capture and operating-system input APIs. It will not add anti-cheat bypasses, memory modification, packet injection, stealth, or detection-evasion features.
