"""App shell for the spectroscopy frontend mockup -- notebook + log bar,
same layout skeleton as am5/gui/app.py (log packed before the notebook so
a tall tab can't squeeze it to nothing, see that file's own comment for
the incident this avoids). Most panels are still a visual mock (see
gui/panels.py's own module docstring for exactly which), but the mount
connection + manual jog control is real: it owns a real
am5.gui.worker.MountWorker, unchanged from the ISS tracker, talking to
either a MockMount or real serial hardware -- same event-pump idiom as
am5/gui/app.py's own _pump_events (EVENT_POLL_MS below), and the same
floating-JogWindow-shown-not-destroyed pattern (see gui/jog_window.py).
"""

from __future__ import annotations

import queue
import tkinter as tk
from datetime import datetime
from tkinter import ttk

from am5.gui.theme import apply_dark_theme
from am5.gui.worker import MountWorker
from spectro.gui.jog_window import JogWindow
from spectro.gui.panels import (
    AcquisitionPanel,
    AlignmentPanel,
    ConnectionPanel,
    FlatsPanel,
    ReductionPanel,
    SpectrumPanel,
    TargetPanel,
)

EVENT_POLL_MS = 100


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Spectro -- Star Analyser capture & reduction")
        root.geometry("1180x760")
        self.palette = apply_dark_theme(root)

        self.mount_worker = MountWorker()

        self.notebook = ttk.Notebook(root)

        self.connection_panel = ConnectionPanel(
            self.notebook, mount_worker=self.mount_worker, on_connection_change=self._on_connection_change,
        )
        # Once per session, not once per star -- measures how the Star
        # Analyser is physically rotated relative to the sensor, shared
        # by both AcquisitionPanel tabs below and by ReductionPanel's own
        # rotation-correction step, see AlignmentPanel's own docstring.
        self.alignment_panel = AlignmentPanel(self.notebook, connection_panel=self.connection_panel)
        self.target_panel = TargetPanel(self.notebook)
        # Reference/Target: two acquisition tabs, not one shared Capture +
        # one shared Calibration tab -- darks must match each capture's
        # own exposure/gain, which can differ between a bright standard
        # and a fainter target, see AcquisitionPanel's own docstring.
        self.reference_panel = AcquisitionPanel(
            self.notebook, role="reference", seed=7, get_star=self.target_panel.get_reference_star,
            get_spectrum=self.target_panel.get_reference_spectrum, mount_worker=self.mount_worker,
            connection_panel=self.connection_panel, alignment_panel=self.alignment_panel,
        )
        self.target_capture_panel = AcquisitionPanel(
            self.notebook, role="target", seed=11, get_star=self.target_panel.get_target_star,
            get_spectrum=self.target_panel.get_target_spectrum_model, mount_worker=self.mount_worker,
            connection_panel=self.connection_panel, alignment_panel=self.alignment_panel,
        )
        self.flats_panel = FlatsPanel(self.notebook)
        # Reduction runs the REAL pipeline (spectro/reduction.py) against
        # whatever's been captured in the three tabs above -- stacking,
        # dark/flat calibration, line detection + dispersion fit, and the
        # reference->target flux calibration. Spectrum then just displays
        # whatever ReductionPanel has computed, see both panels' own
        # docstrings.
        self.reduction_panel = ReductionPanel(
            self.notebook, reference_panel=self.reference_panel, target_capture_panel=self.target_capture_panel,
            flats_panel=self.flats_panel, target_panel=self.target_panel, connection_panel=self.connection_panel,
            alignment_panel=self.alignment_panel,
        )
        self.spectrum_panel = SpectrumPanel(
            self.notebook, reduction_panel=self.reduction_panel, target_panel=self.target_panel,
            connection_panel=self.connection_panel,
        )

        self.notebook.add(self.connection_panel, text="◆ Connection")
        self.notebook.add(self.alignment_panel, text="↗ Alignment")
        self.notebook.add(self.target_panel, text="★ Target & standard")
        self.notebook.add(self.reference_panel, text="⊙ Reference star")
        self.notebook.add(self.target_capture_panel, text="◎ Target")
        self.notebook.add(self.flats_panel, text="▦ Flats")
        self.notebook.add(self.reduction_panel, text="Σ Reduction")
        self.notebook.add(self.spectrum_panel, text="∿ Spectrum")

        # Floating, reachable regardless of the active tab -- created once
        # and hidden/shown (not destroyed) via the button below, see
        # JogWindow.protocol("WM_DELETE_WINDOW", ...). Same reasoning as
        # am5/gui/app.py's own copy of this for packing this BEFORE the
        # notebook despite being created after it: pack() gives space
        # priority in packing ORDER, not creation order, and the
        # notebook's expand=True makes it greedy for any spare height.
        self.jog_window = JogWindow(root, self.mount_worker)
        self.jog_window.withdraw()

        jog_button_frame = ttk.Frame(root)
        jog_button_frame.pack(fill="x", side="bottom")
        ttk.Button(jog_button_frame, text="◆ Jog control...", command=self._show_jog_window).pack(
            anchor="w", padx=10, pady=4,
        )

        log_frame = ttk.LabelFrame(root, text="Log", padding=(6, 2))
        log_frame.pack(fill="x", side="bottom", padx=6, pady=(4, 6))
        log_frame.pack_propagate(False)
        log_frame.configure(height=90)
        self._log_text = tk.Text(
            log_frame, height=4, wrap="word", state="disabled", borderwidth=0,
            background=self.palette.bg_widget, foreground=self.palette.fg_dim,
            insertbackground=self.palette.fg, font=("monospace", 9), padx=6, pady=4,
        )
        self._log_text.pack(fill="both", expand=True)

        self.notebook.pack(fill="both", expand=True, padx=6, pady=(6, 0))

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(EVENT_POLL_MS, self._pump_events)

        self._log("Frontend mockup -- mount connection + jog control (see \"Jog control...\" below) are real, the rest isn't yet.")

    def _pump_events(self) -> None:
        while True:
            try:
                event = self.mount_worker.events.get_nowait()
            except queue.Empty:
                break
            if event.kind == "log":
                self._log(event.payload.get("message", ""))
            else:
                self.connection_panel.handle_mount_event(event)
                self.reference_panel.handle_mount_event(event)
                self.target_capture_panel.handle_mount_event(event)
                self.jog_window.handle_mount_event(event)
        self.root.after(EVENT_POLL_MS, self._pump_events)

    def _show_jog_window(self) -> None:
        self.jog_window.deiconify()
        self.jog_window.lift()
        self.jog_window.focus_force()

    def _on_connection_change(self, connected: bool) -> None:
        self.jog_window.set_connected(connected)

    def _on_close(self) -> None:
        self.mount_worker.shutdown()
        self.root.destroy()

    def _log(self, message: str) -> None:
        self._log_text.configure(state="normal")
        self._log_text.insert("end", f"{datetime.now().strftime('%H:%M:%S')}  {message}\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")


def run() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()
