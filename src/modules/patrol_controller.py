"""Simple full-map patrol/combat state machine for the classic client.

Safety rule: this controller never sends a key unless the captured MapleStory
window is currently the foreground window.  It also uses short key pulses rather
than long holds so focus changes cannot leave movement keys stuck.
"""

from __future__ import annotations

import ctypes
import math
import threading
import time
from typing import Optional, Tuple

from src.common import config
from src.common.vkeys import key_down, key_up, press, release_all


Point = Tuple[int, int]


class PatrolController:
    LOOP_HZ = 30
    LOOP_INTERVAL = 1.0 / LOOP_HZ

    ATTACK_KEY = "shift"
    JUMP_KEY = "space"
    LOOT_KEY = "z"

    ATTACK_INTERVAL = 0.16
    LOOT_INTERVAL = 0.12
    WALK_PULSE = 0.075
    WALK_GAP = 0.015
    FACE_PULSE = 0.035

    MONSTER_CLEAR_GRACE = 0.42
    COMBAT_REACQUIRE_GRACE = 0.90
    KNOCKBACK_PIXEL_JUMP = 90.0
    FAR_MONSTER_DISTANCE = 260.0

    EDGE_LEFT = 0.10
    EDGE_RIGHT = 0.90
    MAP_TOP = 0.18
    MAP_BOTTOM = 0.82
    LEVEL_CHANGE_THRESHOLD = 0.025
    LADDER_ALIGN_PIXELS = 42
    LADDER_SEARCH_PIXELS = 300
    VERTICAL_COOLDOWN = 1.0

    def __init__(self):
        config.patrol_controller = self
        self.ready = False
        self.enabled = True
        self.thread = threading.Thread(target=self._main, name="patrol-controller", daemon=True)
        self._stop_event = threading.Event()

        self._patrol_direction = "right"
        self._vertical_preference = "up"
        self._last_attack = 0.0
        self._last_loot = 0.0
        self._last_monster_seen = 0.0
        self._combat_lock_until = 0.0
        self._previous_monster_distance: Optional[float] = None
        self._last_vertical_action = 0.0
        self._last_focus_warning = 0.0
        self._state = "idle"

    def start(self):
        print("\n[~] Started patrol/combat controller")
        print("[~] Attack=Shift | Jump=Space | Drop=Down+Space | Loot=rapid Z")
        print("[~] Patrol input is blocked unless MapleStory is the foreground window")
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        release_all()

    @staticmethod
    def _foreground_is_game() -> bool:
        capture = getattr(config, "capture", None)
        if capture is None:
            return False
        handle = int(getattr(capture, "_handle", 0) or 0)
        if not handle:
            return False
        try:
            foreground = int(ctypes.windll.user32.GetForegroundWindow() or 0)
            return foreground == handle and bool(ctypes.windll.user32.IsWindow(handle))
        except Exception:
            return False

    def _safe_press(self, key: str, down_time: float = 0.04, up_time: float = 0.02) -> bool:
        if not config.enabled or not self._foreground_is_game():
            return False
        press(key, 1, down_time=down_time, up_time=up_time)
        return True

    def _safe_combo(self, first: str, second: str, first_lead: float = 0.025, hold: float = 0.08) -> bool:
        if not config.enabled or not self._foreground_is_game():
            return False
        key_down(first)
        try:
            time.sleep(first_lead)
            if not self._foreground_is_game():
                return False
            key_down(second)
            try:
                time.sleep(hold)
            finally:
                key_up(second)
        finally:
            key_up(first)
        return True

    @staticmethod
    def _player_anchor() -> Optional[Point]:
        capture = getattr(config, "capture", None)
        if capture is None:
            return None
        with capture._state_lock:
            frame = capture.frame
            if frame is None:
                return None
            height, width = frame.shape[:2]
        # The classic camera keeps the local character near screen center.  This
        # is the current character anchor until the scene model gains a dedicated
        # player class; all monster ranges are measured from this one point.
        return width // 2, height // 2

    @staticmethod
    def _minimap_player() -> Optional[Tuple[float, float]]:
        capture = getattr(config, "capture", None)
        if capture is None or not getattr(capture, "player_found", False):
            return None
        with capture._state_lock:
            pos = getattr(capture, "_filtered_position", None)
            return None if pos is None else (float(pos[0]), float(pos[1]))

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _nearest_monster(self, snapshot, anchor: Point):
        monsters = tuple(getattr(snapshot, "monsters", ()) or ())
        if not monsters:
            return None, None
        target = min(monsters, key=lambda item: self._distance(anchor, item.center))
        return target, self._distance(anchor, target.center)

    def _face_target(self, anchor: Point, monster) -> None:
        if monster is None:
            return
        direction = "right" if monster.center[0] >= anchor[0] else "left"
        self._safe_press(direction, down_time=self.FACE_PULSE, up_time=0.005)

    def _combat(self, snapshot, anchor: Point, now: float) -> bool:
        monster, distance = self._nearest_monster(snapshot, anchor)
        if monster is not None:
            self._last_monster_seen = now
            self._combat_lock_until = max(self._combat_lock_until, now + self.MONSTER_CLEAR_GRACE)

            if self._previous_monster_distance is not None and distance is not None:
                jumped = distance - self._previous_monster_distance
                if jumped >= self.KNOCKBACK_PIXEL_JUMP:
                    # A sudden increase usually means the character was knocked
                    # back. Keep combat locked instead of incorrectly resuming patrol.
                    self._combat_lock_until = max(
                        self._combat_lock_until, now + self.COMBAT_REACQUIRE_GRACE
                    )
                    self._state = "combat-knockback-recovery"

            self._previous_monster_distance = distance
            self._face_target(anchor, monster)

            if distance is not None and distance > self.FAR_MONSTER_DISTANCE:
                direction = "right" if monster.center[0] >= anchor[0] else "left"
                self._safe_press(direction, down_time=0.055, up_time=0.005)

            if now - self._last_attack >= self.ATTACK_INTERVAL:
                self._safe_press(self.ATTACK_KEY, down_time=0.055, up_time=0.025)
                self._last_attack = now
            self._state = "combat"
            return True

        # The target can briefly disappear behind hit effects or during knockback.
        # Continue swinging for a short grace period before patrol resumes.
        if now < self._combat_lock_until:
            if now - self._last_attack >= self.ATTACK_INTERVAL:
                self._safe_press(self.ATTACK_KEY, down_time=0.055, up_time=0.025)
                self._last_attack = now
            self._state = "combat-reacquire"
            return True

        self._previous_monster_distance = None
        return False

    def _rapid_loot(self, now: float) -> None:
        if now - self._last_loot < self.LOOT_INTERVAL:
            return
        # Deliberately repeated taps, never a held Z key.
        self._safe_press(self.LOOT_KEY, down_time=0.025, up_time=0.025)
        self._last_loot = now

    def _nearest_ladder(self, snapshot, anchor: Point):
        ladders = tuple(getattr(snapshot, "ladders", ()) or ())
        if not ladders:
            return None
        candidates = [
            ladder for ladder in ladders
            if abs(ladder.center[0] - anchor[0]) <= self.LADDER_SEARCH_PIXELS
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(item.center[0] - anchor[0]))

    @staticmethod
    def _platform_below(snapshot, anchor: Point) -> bool:
        for platform in tuple(getattr(snapshot, "platforms", ()) or ()):
            x1, y1, x2, _y2 = platform.box
            if y1 <= anchor[1] + 35:
                continue
            if x1 - 50 <= anchor[0] <= x2 + 50:
                return True
        return False

    def _climb_ladder(self, ladder, anchor: Point) -> bool:
        dx = ladder.center[0] - anchor[0]
        if abs(dx) > self.LADDER_ALIGN_PIXELS:
            direction = "right" if dx > 0 else "left"
            self._safe_press(direction, down_time=0.09, up_time=0.01)
            self._state = "align-ladder"
            return False

        before = self._minimap_player()
        self._state = "climb-ladder"
        # User-specified sequence: jump, then immediately Up to catch ladder.
        if not self._safe_combo(self.JUMP_KEY, "up", first_lead=0.025, hold=0.11):
            return False
        for _ in range(5):
            if not config.enabled or not self._foreground_is_game():
                break
            self._safe_press("up", down_time=0.10, up_time=0.025)
        time.sleep(0.10)
        after = self._minimap_player()
        if before is not None and after is not None:
            return before[1] - after[1] >= self.LEVEL_CHANGE_THRESHOLD
        return True

    def _drop_down(self) -> bool:
        before = self._minimap_player()
        self._state = "drop-down"
        if not self._safe_combo("down", self.JUMP_KEY, first_lead=0.03, hold=0.08):
            return False
        time.sleep(0.22)
        after = self._minimap_player()
        if before is not None and after is not None:
            return after[1] - before[1] >= self.LEVEL_CHANGE_THRESHOLD
        return True

    def _edge_transition(self, snapshot, anchor: Point, now: float, pos) -> bool:
        if now - self._last_vertical_action < self.VERTICAL_COOLDOWN:
            return False

        x, y = pos
        at_edge = x <= self.EDGE_LEFT or x >= self.EDGE_RIGHT
        if not at_edge:
            return False

        if y <= self.MAP_TOP:
            self._vertical_preference = "down"
        elif y >= self.MAP_BOTTOM:
            self._vertical_preference = "up"

        changed = False
        ladder = self._nearest_ladder(snapshot, anchor)

        if self._vertical_preference == "up" and ladder is not None:
            changed = self._climb_ladder(ladder, anchor)
        elif self._vertical_preference == "down" and self._platform_below(snapshot, anchor):
            changed = self._drop_down()
        elif ladder is not None:
            changed = self._climb_ladder(ladder, anchor)
        elif self._platform_below(snapshot, anchor):
            changed = self._drop_down()

        self._last_vertical_action = now
        if changed:
            self._patrol_direction = "left" if self._patrol_direction == "right" else "right"
            self._state = "level-changed"
            return True

        # No usable vertical route at this edge: turn around instead of getting stuck.
        self._patrol_direction = "left" if self._patrol_direction == "right" else "right"
        self._state = "edge-turnaround"
        return True

    def _patrol(self, snapshot, anchor: Point, now: float) -> None:
        pos = self._minimap_player()
        if pos is not None and self._edge_transition(snapshot, anchor, now, pos):
            return

        self._safe_press(
            self._patrol_direction,
            down_time=self.WALK_PULSE,
            up_time=self.WALK_GAP,
        )
        self._state = f"patrol-{self._patrol_direction}"

    def _main(self):
        self.ready = True
        while not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                if not config.enabled:
                    self._state = "idle"
                    self._previous_monster_distance = None
                    time.sleep(0.03)
                    continue

                if not self._foreground_is_game():
                    # Do not send anything to Desktop, browser, CMD, or any other app.
                    release_all()
                    now = time.monotonic()
                    if now - self._last_focus_warning >= 3.0:
                        print("\n[~] Patrol paused: MapleStory is not the foreground window")
                        self._last_focus_warning = now
                    time.sleep(0.08)
                    continue

                observer = getattr(config, "scene_observer", None)
                if observer is None:
                    time.sleep(0.05)
                    continue

                snapshot = observer.snapshot()
                anchor = self._player_anchor()
                if anchor is None:
                    time.sleep(0.05)
                    continue

                now = time.monotonic()
                self._rapid_loot(now)
                if not self._combat(snapshot, anchor, now):
                    self._patrol(snapshot, anchor, now)

            except Exception as exc:
                release_all()
                self._state = "error"
                print(f"\n[!] Patrol controller error: {type(exc).__name__}: {exc}")
                time.sleep(0.25)

            remaining = self.LOOP_INTERVAL - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
