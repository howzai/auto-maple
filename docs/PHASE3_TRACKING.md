# Phase 3 — Player tracking stabilization

## Implemented

- Selects the player candidate nearest to the previous accepted position.
- Uses template confidence only as a tie-breaker instead of blindly taking the first match.
- Applies exponential moving-average smoothing to normal movement.
- Rejects isolated large coordinate jumps.
- Accepts portal/teleport movement only after several consecutive frames agree.
- Deduplicates overlapping template matches.
- Publishes player confidence, candidate count, frame age, rejected jumps, and capture health.
- Adds a watchdog that pauses automation if the capture worker dies or frames stall.

## Current defaults

- Position EMA alpha: `0.45`
- Maximum normal normalized jump: `0.18`
- Teleport confirmation: `3` frames
- Teleport cluster radius: `0.08`
- Capture stall timeout: `2.5` seconds

These values require Windows validation on multiple maps and resolutions before merging.
