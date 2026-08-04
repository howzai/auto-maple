import tkinter as tk

from src.common import config
from src.gui.interfaces import LabelFrame


STATUS_LABELS = {
    'healthy': 'Healthy',
    'waiting-for-window': 'Waiting for game window',
    'calibrating': 'Calibrating minimap',
    'waiting-for-player': 'Waiting for player marker',
    'capture-stopped': 'Capture stopped',
    'error': 'Error',
}


class Status(LabelFrame):
    """Display routine metadata and live runtime diagnostics."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, 'Status', **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)

        self.curr_cb = tk.StringVar()
        self.curr_routine = tk.StringVar()
        self.runtime_state = tk.StringVar(value='Starting')
        self.capture_fps = tk.StringVar(value='0.0')
        self.player_confidence = tk.StringVar(value='0%')
        self.player_position = tk.StringVar(value='—')
        self.frame_age = tk.StringVar(value='—')
        self.scene_model = tk.StringVar(value='Starting')
        self.scene_counts = tk.StringVar(value='M 0 · L 0 · O 0')
        self.nearest_monster = tk.StringVar(value='None')
        self.scene_event = tk.StringVar(value='Waiting')
        self.recorder_state = tk.StringVar(value='Idle')
        self.recorder_counts = tk.StringVar(value='S 0 · F 0 · E 0 · 0 MB')
        self.recorder_event = tk.StringVar(value='Idle')
        self.last_error = tk.StringVar(value='None')

        self._add_row(0, 'Command Book:', self.curr_cb)
        self._add_row(1, 'Routine:', self.curr_routine)
        self._add_row(2, 'Runtime:', self.runtime_state)
        self._add_row(3, 'Capture FPS:', self.capture_fps)
        self._add_row(4, 'Player confidence:', self.player_confidence)
        self._add_row(5, 'Player position:', self.player_position)
        self._add_row(6, 'Frame age:', self.frame_age)
        self._add_row(7, 'Scene model:', self.scene_model)
        self._add_row(8, 'Scene objects:', self.scene_counts)
        self._add_row(9, 'Nearest monster:', self.nearest_monster)
        self._add_row(10, 'Vision event:', self.scene_event, wrap=True)
        self._add_row(11, 'Recorder:', self.recorder_state)
        self._add_row(12, 'Recorded data:', self.recorder_counts)
        self._add_row(13, 'Recorder report:', self.recorder_event, wrap=True)
        self._add_row(14, 'Last error:', self.last_error, wrap=True)

    def _add_row(self, row, label_text, variable, wrap=False):
        label = tk.Label(self, text=label_text)
        label.grid(row=row, column=1, padx=5, pady=2, sticky=tk.E)
        if wrap:
            value = tk.Label(self, textvariable=variable, anchor=tk.W, justify=tk.LEFT,
                             wraplength=360, relief=tk.SUNKEN, borderwidth=1)
        else:
            value = tk.Entry(self, textvariable=variable, state=tk.DISABLED)
        value.grid(row=row, column=2, padx=(0, 5), pady=2, sticky=tk.EW)

    def set_cb(self, string):
        self.curr_cb.set(string)

    def set_routine(self, string):
        self.curr_routine.set(string)

    def update_health(self, health):
        capture = health.capture
        state = STATUS_LABELS.get(health.status, health.status)
        mode = 'Enabled' if health.enabled else 'Paused'
        self.runtime_state.set(f'{state} · {mode}')
        self.capture_fps.set(f'{capture.fps:.1f}')
        self.player_confidence.set(f'{capture.player_confidence * 100:.0f}%')
        if capture.player_found:
            x, y = capture.player_position
            self.player_position.set(f'({x:.3f}, {y:.3f})')
        else:
            self.player_position.set('Not detected')
        self.frame_age.set('No frame' if health.capture_frame_age is None else f'{health.capture_frame_age:.2f} s')

        observer = getattr(config, 'scene_observer', None)
        scene = observer.snapshot() if observer is not None else None
        if scene is None:
            self.scene_model.set('Unavailable')
            self.scene_counts.set('M 0 · L 0 · O 0')
            self.nearest_monster.set('None')
            self.scene_event.set('Scene observer unavailable')
        else:
            debug_suffix = ' · Debug' if getattr(observer, 'debug_enabled', False) else ''
            self.scene_model.set(f'{scene.model_status}{debug_suffix}')
            self.scene_counts.set(f'M {len(scene.monsters)} · L {len(scene.ladders)} · O {len(scene.obstacles)}')
            if scene.nearest_monster_distance is None:
                self.nearest_monster.set('None')
            else:
                self.nearest_monster.set(f'{scene.nearest_monster_direction} · {scene.nearest_monster_distance:.0f}px')
            self.scene_event.set(scene.last_event)

        recorder = getattr(config, 'data_recorder', None)
        recording = recorder.snapshot() if recorder is not None else None
        if recording is None:
            self.recorder_state.set('Unavailable')
            self.recorder_counts.set('S 0 · F 0 · E 0 · 0 MB')
            self.recorder_event.set('Recorder unavailable')
        else:
            if recording.recording:
                minutes, seconds = divmod(int(recording.elapsed_seconds), 60)
                self.recorder_state.set(
                    f'RECORDING · {minutes:02d}:{seconds:02d} · Smart'
                )
                self.recorder_event.set(recording.last_event)
            else:
                self.recorder_state.set('Idle · Smart')
                if recording.readiness_score:
                    self.recorder_event.set(
                        f'Quality {recording.quality_score}% · AI {recording.readiness_score}% · '
                        f'{recording.recommendation}'
                    )
                else:
                    self.recorder_event.set(recording.last_event)
            self.recorder_counts.set(
                f'S {recording.samples} · F {recording.frames_saved} · '
                f'E {recording.events} · {recording.estimated_size_mb:.1f} MB'
            )

        errors = [capture.last_error]
        if scene is not None:
            errors.append(scene.last_error)
        if recording is not None:
            errors.append(recording.last_error)
        self.last_error.set(next((error for error in errors if error), 'None'))
