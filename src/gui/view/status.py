import tkinter as tk

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
        self.last_error = tk.StringVar(value='None')

        self._add_row(0, 'Command Book:', self.curr_cb)
        self._add_row(1, 'Routine:', self.curr_routine)
        self._add_row(2, 'Runtime:', self.runtime_state)
        self._add_row(3, 'Capture FPS:', self.capture_fps)
        self._add_row(4, 'Player confidence:', self.player_confidence)
        self._add_row(5, 'Player position:', self.player_position)
        self._add_row(6, 'Frame age:', self.frame_age)
        self._add_row(7, 'Last error:', self.last_error, wrap=True)

    def _add_row(self, row, label_text, variable, wrap=False):
        label = tk.Label(self, text=label_text)
        label.grid(row=row, column=1, padx=5, pady=2, sticky=tk.E)

        if wrap:
            value = tk.Label(
                self,
                textvariable=variable,
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=360,
                relief=tk.SUNKEN,
                borderwidth=1,
            )
        else:
            value = tk.Entry(self, textvariable=variable, state=tk.DISABLED)
        value.grid(row=row, column=2, padx=(0, 5), pady=2, sticky=tk.EW)

    def set_cb(self, string):
        self.curr_cb.set(string)

    def set_routine(self, string):
        self.curr_routine.set(string)

    def update_health(self, health):
        """Render a RuntimeHealthSnapshot on the Tk main thread."""
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

        if health.capture_frame_age is None:
            self.frame_age.set('No frame')
        else:
            self.frame_age.set(f'{health.capture_frame_age:.2f} s')

        self.last_error.set(capture.last_error or 'None')
