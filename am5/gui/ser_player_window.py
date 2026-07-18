"""Floating SER-file viewer window -- same "created once, shown/hidden on
demand" pattern as JogWindow/FinderWindow (see am5/gui/app.py, which owns
the one instance).

Moved out of the main Notebook (was the "SER player" tab) since a plain
tab's position in a wrapping multi-row tab strip isn't stable -- with
enough other tabs, it could land at the START of a wrapped second row
(visually top-left, the opposite of the "separate, standalone tool" feel
it's meant to have) instead of consistently off to the side the way a
Toplevel button always is. SerPlayerPanel itself (am5/gui/panels.py) is
unchanged -- this is a thin wrapper, not a rewrite: pure local file I/O,
no worker/device involved, so it needs no event wiring here either.
"""

from __future__ import annotations

import tkinter as tk

from am5.gui.panels import SerPlayerPanel
from am5.gui.theme import PALETTE


class SerPlayerWindow(tk.Toplevel):
    def __init__(self, parent: tk.Misc):
        super().__init__(parent)
        self.title("SER player")
        # Toplevel is a raw tk widget -- apply_dark_theme's ttk.Style
        # doesn't reach it (see am5/gui/theme.py's docstring).
        self.configure(background=PALETTE.bg)
        self._panel = SerPlayerPanel(self)
        self._panel.pack(fill="both", expand=True)

        # Closing the window (X button) just hides it -- App keeps the one
        # instance alive so it can be reopened without losing state (the
        # currently-loaded file, playback position), same as JogWindow/
        # FinderWindow.
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
