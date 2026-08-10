"""Performance tuning for the combined scene observer.

The classic client should remain responsive while Auto Maple is idle and while
patrol is active. This patch keeps WGC running, but avoids expensive YOLO work
until automation or F10 debug is actually enabled.
"""

from __future__ import annotations

import time

from src.common import config


def install_scene_performance_patch(scene_observer_class) -> None:
    if getattr(scene_observer_class, "_performance_patch_installed", False):
        return

    scene_observer_class.LOOP_HZ = 30
    scene_observer_class.LOOP_INTERVAL = 1.0 / 30.0
    scene_observer_class.MONSTER_HZ = 10
    scene_observer_class.MONSTER_INTERVAL = 1.0 / 10.0
    scene_observer_class.NAVIGATION_HZ = 1
    scene_observer_class.NAVIGATION_INTERVAL = 1.0
    scene_observer_class.DEBUG_HZ = 5
    scene_observer_class.DEBUG_INTERVAL = 1.0 / 5.0
    scene_observer_class.IMAGE_SIZE = 480

    original_observe_once = scene_observer_class._observe_once

    def observe_only_when_needed(self):
        # WGC/minimap tracking remain live, but main-scene YOLO sleeps while the
        # bot is idle. F10 can still explicitly wake it for diagnostics.
        if not config.enabled and not self.debug_enabled:
            return
        return original_observe_once(self)

    @staticmethod
    def predict_optimized(model, frame, confidence, device):
        bgr = frame[:, :, :3]
        started = time.perf_counter()
        # Do not pass Ultralytics' deprecated ``half`` argument.  The lower
        # inference size/rates provide the primary performance win and avoid
        # flooding the console with a warning on every prediction.
        results = model.predict(
            bgr,
            imgsz=scene_observer_class.IMAGE_SIZE,
            conf=confidence,
            device=device,
            verbose=False,
        )
        return results, (time.perf_counter() - started) * 1000.0

    scene_observer_class._observe_once = observe_only_when_needed
    scene_observer_class._predict_model = predict_optimized
    scene_observer_class._performance_patch_installed = True
