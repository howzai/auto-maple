# Phase 2 summary

The capture loop now behaves as a service that can wait safely for the game instead of assuming the game is already open. Unsafe capture states disable automation and release held input. Minimap and player detection now expose explicit health state used by the listener before enabling a routine.
