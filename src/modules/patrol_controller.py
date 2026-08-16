"""Full-map patrol/combat state machine for the classic client.

Patrol movement is continuous, loot is held, combat only engages nearby monsters
in the current sweep direction, wall/edge handling searches for the nearest
ladder, and sudden minimap displacement is treated as knockback so combat can
recover instead of stalling.
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

    ATTACK_KEY = "a"
    JUMP_KEY = "space"
    LOOT_KEY = "z"

    ATTACK_INTERVAL = 0.16
    ATTACK_DISTANCE = 210.0
    CHASE_DISTANCE = 340.0
    MONSTER_VERTICAL_TOLERANCE = 90
    FORWARD_DEADZONE = 10

    KNOCKBACK_PIXEL_JUMP = 90.0
    MINIMAP_KNOCKBACK_X = 0.055
    KNOCKBACK_RECOVERY_SECONDS = 0.90

    EDGE_LEFT = 0.12
    EDGE_RIGHT = 0.88
    MAP_TOP = 0.18
    MAP_BOTTOM = 0.82
    LEVEL_CHANGE_THRESHOLD = 0.025
    LADDER_ALIGN_PIXELS = 42
    LADDER_SEARCH_PIXELS = 500
    VERTICAL_COOLDOWN = 0.85

    def __init__(self):
        config.patrol_controller = self
        self.ready = False
        self.enabled = True
        self.thread = threading.Thread(target=self._main, name="patrol-controller", daemon=True)
        self._stop_event = threading.Event()

        self._patrol_direction = "right"
        self._vertical_preference = "up"
        self._last_attack = 0.0
        self._previous_monster_distance: Optional[float] = None
        self._last_vertical_action = 0.0
        self._last_focus_warning = 0.0
        self._state = "idle"

        self._held_direction: Optional[str] = None
        self._loot_held = False
        self._searching_ladder = False
        self._last_minimap_pos: Optional[Tuple[float, float]] = None
        self._last_target_direction: Optional[str] = None
        self._knockback_recover_until = 0.0

    def start(self):
        print("\n[~] Started patrol/combat controller")
        print("[~] Attack=A | Jump=Space | Drop=Space+Down | Climb=Space+Up | Loot=hold Z")
        print("[~] Nearby forward monsters only; knockback uses minimap recovery")
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        self._release_motion()
        self._release_loot()
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

    @staticmethod
    def _focus_debug_text() -> str:
        return "focus diagnostics unavailable"

    def _safe_press(self, key: str, down_time: float = 0.04, up_time: float = 0.02) -> bool:
        if not config.enabled or not self._foreground_is_game():
            return False
        press(key, 1, down_time=down_time, up_time=up_time)
        return True

    def _safe_key_down(self, key: str) -> bool:
        if not config.enabled or not self._foreground_is_game():
            return False
        key_down(key)
        return True

    def _safe_key_up(self, key: str) -> bool:
        try:
            key_up(key)
            return True
        except Exception:
            return False

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

    def _set_motion(self, direction: Optional[str]) -> None:
        if direction == self._held_direction:
            return
        if self._held_direction is not None:
            self._safe_key_up(self._held_direction)
            self._held_direction = None
        if direction is not None and self._safe_key_down(direction):
            self._held_direction = direction

    def _release_motion(self) -> None:
        if self._held_direction is not None:
            self._safe_key_up(self._held_direction)
            self._held_direction = None

    def _ensure_loot_held(self) -> None:
        if not self._loot_held and self._safe_key_down(self.LOOT_KEY):
            self._loot_held = True

    def _release_loot(self) -> None:
        if self._loot_held:
            self._safe_key_up(self.LOOT_KEY)
            self._loot_held = False

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

    def _detect_minimap_knockback(self, pos: Optional[Tuple[float, float]], now: float) -> bool:
        previous = self._last_minimap_pos
        self._last_minimap_pos = pos
        if pos is None or previous is None:
            return False
        if abs(pos[0] - previous[0]) >= self.MINIMAP_KNOCKBACK_X:
            self._knockback_recover_until = max(
                self._knockback_recover_until,
                now + self.KNOCKBACK_RECOVERY_SECONDS,
            )
            self._state = "combat-knockback-recovery"
            return True
        return False

    def _all_monsters(self, snapshot):
        return tuple(getattr(snapshot, "monsters", ()) or ())

    def _nearest_monster(self, snapshot, anchor: Point):
        monsters = self._all_monsters(snapshot)
        if not monsters:
            return None, None
        target = min(monsters, key=lambda item: self._distance(anchor, item.center))
        return target, self._distance(anchor, target.center)

    def _forward_monster(self, snapshot, anchor: Point):
        candidates = []
        for monster in self._all_monsters(snapshot):
            dx = monster.center[0] - anchor[0]
            dy = abs(monster.center[1] - anchor[1])
            if dy > self.MONSTER_VERTICAL_TOLERANCE:
                continue
            if self._patrol_direction == "right" and dx < self.FORWARD_DEADZONE:
                continue
            if self._patrol_direction == "left" and dx > -self.FORWARD_DEADZONE:
                continue
            distance = self._distance(anchor, monster.center)
            if distance <= self.CHASE_DISTANCE:
                candidates.append((distance, monster))
        if not candidates:
            return None, None
        distance, target = min(candidates, key=lambda item: item[0])
        return target, distance

    def _attack_if_ready(self, now: float) -> None:
        if now - self._last_attack >= self.ATTACK_INTERVAL:
            self._safe_press(self.ATTACK_KEY, down_time=0.055, up_time=0.025)
            self._last_attack = now

    def _recover_from_knockback(self, snapshot, anchor: Point, now: float) -> bool:
        if now >= self._knockback_recover_until:
            return False

        monster, distance = self._nearest_monster(snapshot, anchor)
        if monster is not None:
            direction = "right" if monster.center[0] >= anchor[0] else "left"
            self._last_target_direction = direction
            if distance is not None and distance > self.ATTACK_DISTANCE:
                self._set_motion(direction)
                self._state = "combat-knockback-chase"
            else:
                self._release_motion()
                self._attack_if_ready(now)
                self._state = "combat-knockback-attack"
            return True

        if self._last_target_direction is not None:
            self._set_motion(self._last_target_direction)
            self._state = "combat-knockback-reacquire"
            return True
        return False

    def _combat(self, snapshot, anchor: Point, now: float) -> bool:
        if self._recover_from_knockback(snapshot, anchor, now):
            return True

        monster, distance = self._forward_monster(snapshot, anchor)
        if monster is None:
            self._previous_monster_distance = None
            return False

        direction = "right" if monster.center[0] >= anchor[0] else "left"
        self._last_target_direction = direction

        if self._previous_monster_distance is not None and distance is not None:
            if distance - self._previous_monster_distance >= self.KNOCKBACK_PIXEL_JUMP:
                self._knockback_recover_until = max(
                    self._knockback_recover_until,
                    now + self.KNOCKBACK_RECOVERY_SECONDS,
                )
        self._previous_monster_distance = distance

        if distance is not None and distance > self.ATTACK_DISTANCE:
            self._set_motion(direction)
            self._state = "chase-forward-monster"
            return True

        self._release_motion()
        self._attack_if_ready(now)
        self._state = f"combat-{self._patrol_direction}"
        return True

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
            self._set_motion("right" if dx > 0 else "left")
            self._state = "align-ladder"
            return False

        self._release_motion()
        before = self._minimap_player()
        self._state = "climb-ladder"
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
        self._release_motion()
        before = self._minimap_player()
        self._state = "drop-down"
        if not self._safe_combo(self.JUMP_KEY, "down", first_lead=0.03, hold=0.08):
            return False
        time.sleep(0.22)
        after = self._minimap_player()
        if before is not None and after is not None:
            return after[1] - before[1] >= self.LEVEL_CHANGE_THRESHOLD
        return True

    def _reverse_patrol_direction(self) -> None:
        self._patrol_direction = "left" if self._patrol_direction == "right" else "right"

    def _search_for_ladder(self, snapshot, anchor: Point) -> bool:
        ladder = self._nearest_ladder(snapshot, anchor)
        if ladder is None:
            self._set_motion(self._patrol_direction)
            self._state = "search-ladder"
            return True

        if self._climb_ladder(ladder, anchor):
            self._searching_ladder = False
            self._reverse_patrol_direction()
            self._state = "level-up-turnaround"
        return True

    def _edge_transition(self, snapshot, anchor: Point, now: float, pos) -> bool:
        if self._searching_ladder:
            return self._search_for_ladder(snapshot, anchor)
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

        ladder = self._nearest_ladder(snapshot, anchor)

        if self._vertical_preference == "up":
            if ladder is not None:
                changed = self._climb_ladder(ladder, anchor)
                self._last_vertical_action = now
                if changed:
                    self._reverse_patrol_direction()
                    self._state = "level-up-turnaround"
                return True

            self._patrol_direction = "left" if x >= self.EDGE_RIGHT else "right"
            self._searching_ladder = True
            self._set_motion(self._patrol_direction)
            self._state = "reverse-search-ladder"
            self._last_vertical_action = now
            return True

        if self._platform_below(snapshot, anchor):
            changed = self._drop_down()
            self._last_vertical_action = now
            if changed:
                self._reverse_patrol_direction()
                self._state = "level-down-turnaround"
            return True

        self._reverse_patrol_direction()
        self._set_motion(self._patrol_direction)
        self._state = "edge-turnaround"
        self._last_vertical_action = now
        return True

    def _patrol(self, snapshot, anchor: Point, now: float) -> None:
        pos = self._minimap_player()
        if pos is not None and self._edge_transition(snapshot, anchor, now, pos):
            return
        self._set_motion(self._patrol_direction)
        self._state = f"patrol-{self._patrol_direction}"

    def _reset_held_state(self) -> None:
        self._held_direction = None
        self._loot_held = False

    def _main(self):
        self.ready = True
        while not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                if not config.enabled:
                    self._release_motion()
                    self._release_loot()
                    self._state = "idle"
                    self._previous_monster_distance = None
                    self._last_minimap_pos = None
                    time.sleep(0.03)
                    continue

                if not self._foreground_is_game():
                    release_all()
                    self._reset_held_state()
                    now = time.monotonic()
                    if now - self._last_focus_warning >= 3.0:
                        print("\n[~] Patrol paused: MapleStory is not the foreground window")
                        try:
                            print(f"[FOCUS] {self._focus_debug_text()}")
                        except Exception as exc:
                            print(f"[FOCUS] diagnostic failed: {type(exc).__name__}: {exc}")
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
                pos = self._minimap_player()
                self._detect_minimap_knockback(pos, now)
                self._ensure_loot_held()

                if not self._combat(snapshot, anchor, now):
                    self._patrol(snapshot, anchor, now)

            except Exception as exc:
                release_all()
                self._reset_held_state()
                self._state = "error"
                print(f"\n[!] Patrol controller error: {type(exc).__name__}: {exc}")
                time.sleep(0.25)

            remaining = self.LOOP_INTERVAL - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
