"""User-friendly GUI for interacting with Auto Maple.

Tkinter widgets must only be updated from the main GUI thread. Periodic minimap and
runtime-health refreshes therefore use ``after`` instead of background threads.
"""

import threading
import tkinter as tk
from tkinter import ttk

from src.common import config, settings
from src.common.health import runtime_health
from src.gui import Edit, Menu, Settings, View


class GUI:
    DISPLAY_FRAME_RATE = 30
    HEALTH_REFRESH_MS = 250
    RESOLUTIONS = {
        'DEFAULT': '800x860',
        'Edit': '1400x800',
    }

    def __init__(self):
        config.gui = self

        self.root = tk.Tk()
        self.root.title('Auto Maple')
        icon = tk.PhotoImage(file='assets/icon.png')
        self.root.iconphoto(False, icon)
        self.root.geometry(GUI.RESOLUTIONS['DEFAULT'])
        self.root.resizable(False, False)

        self.routine_var = tk.StringVar()
        self._closing = False

        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)

        self.navigation = ttk.Notebook(self.root)
        self.view = View(self.navigation)
        self.edit = Edit(self.navigation)
        self.settings = Settings(self.navigation)

        self.navigation.pack(expand=True, fill='both')
        self.navigation.bind('<<NotebookTabChanged>>', self._resize_window)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self.root.focus()

    def set_routine(self, arr):
        self.routine_var.set(arr)

    def clear_routine_info(self):
        self.view.details.clear_info()
        self.view.status.set_routine('')

        self.edit.minimap.redraw()
        self.edit.routine.commands.clear_contents()
        self.edit.routine.commands.update_display()
        self.edit.editor.reset()

    def _resize_window(self, event):
        nav = event.widget
        curr_id = nav.select()
        nav.nametowidget(curr_id).focus()
        page = nav.tab(curr_id, 'text')
        if self.root.state() != 'zoomed':
            self.root.geometry(GUI.RESOLUTIONS.get(page, GUI.RESOLUTIONS['DEFAULT']))

    def start(self):
        """Start scheduled GUI updates and enter the Tk event loop."""
        self._schedule_minimap_refresh()
        self._schedule_health_refresh()

        layout_thread = threading.Thread(
            target=self._save_layout,
            name='layout-autosave',
            daemon=True,
        )
        layout_thread.start()
        self.root.mainloop()

    def _schedule_minimap_refresh(self):
        if self._closing:
            return
        try:
            self.view.minimap.display_minimap()
        finally:
            delay_ms = max(1, round(1000 / GUI.DISPLAY_FRAME_RATE))
            self.root.after(delay_ms, self._schedule_minimap_refresh)

    def _schedule_health_refresh(self):
        if self._closing:
            return
        try:
            self.view.status.update_health(runtime_health())
        finally:
            self.root.after(GUI.HEALTH_REFRESH_MS, self._schedule_health_refresh)

    def _save_layout(self):
        """Periodically save the current Layout object."""
        while not self._closing:
            if config.layout is not None and settings.record_layout:
                try:
                    config.layout.save()
                except Exception as exc:
                    print(f'\n[!] Failed to save layout: {exc}')
            threading.Event().wait(5)

    def _on_close(self):
        self._closing = True
        config.enabled = False
        capture = getattr(config, 'capture', None)
        if capture is not None and hasattr(capture, 'stop'):
            capture.stop()
        self.root.destroy()


if __name__ == '__main__':
    GUI().start()
