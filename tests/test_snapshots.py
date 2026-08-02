"""Tests for immutable runtime state objects."""

import unittest
from dataclasses import FrozenInstanceError

from src.common.snapshots import CaptureSnapshot


class CaptureSnapshotTests(unittest.TestCase):
    def test_healthy_requires_all_tracking_states(self):
        snapshot = CaptureSnapshot(
            window_found=True,
            calibrated=True,
            player_found=True,
            player_confidence=0.95,
        )
        self.assertTrue(snapshot.healthy)

    def test_error_makes_snapshot_unhealthy(self):
        snapshot = CaptureSnapshot(
            window_found=True,
            calibrated=True,
            player_found=True,
            last_error="capture stalled",
        )
        self.assertFalse(snapshot.healthy)

    def test_snapshot_is_immutable(self):
        snapshot = CaptureSnapshot(frame_id=7)
        with self.assertRaises(FrozenInstanceError):
            snapshot.frame_id = 8


if __name__ == "__main__":
    unittest.main()
