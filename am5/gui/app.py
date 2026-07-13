"""Main GUI window: wires the panels together, pumps MountWorker's and
CameraWorker's event queues into them, and does the small amount of
cross-panel wiring (pass selection -> trajectory, calibration -> axis
signs) that no single panel should own by itself.
"""

from __future__ import annotations

import queue
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk

from am5.gui.jog_window import JogWindow
from am5.gui.panels import (
    CameraControlVars,
    ConnectionPanel,
    ExposurePanel,
    CalibrationPanel,
    PassesPanel,
    SerPlayerPanel,
    SiteVars,
    TransitPanel,
)
from am5.gui.theme import apply_dark_theme
from am5.gui.worker import MountWorker
from am5.tracker import AxisSigns, LiveOffsets
from camera.worker import CameraWorker

# Events after which the operator has just confirmed/reset "on target" --
# the idle-mode camera reference re-acquires from the next position sample
# rather than carrying over a stale offset from before the move.
CAMERA_REFERENCE_RESET_EVENTS = {"connected", "goto_arrived", "parked"}

EVENT_POLL_MS = 100
LOG_EVENT_KINDS = {"log", "connect_error", "tracking_error"}


class App:
    def __init__(self, root: tk.Tk, out_dir: Path):
        self.root = root
        self.worker = MountWorker()
        self.camera_worker = CameraWorker()

        root.title("AM3/AM5 ISS Tracker")
        # Applied once, here, before any widget below is built -- ttk.Style
        # is process-wide, so every panel's ttk widgets pick up the dark
        # theme automatically just by being created after this call.
        self.palette = apply_dark_theme(root)
        self.notebook = ttk.Notebook(root)
        # Not packed yet -- deliberately deferred until after the jog
        # button and log bar below are packed (see the comment there for
        # why the order matters).

        # Shared with CalibrationPanel so a camera-detected correction lands in
        # the same offsets the active tracking loop is reading.
        self.live_offsets = LiveOffsets()
        # Shared with JogWindow (same instance) so a (re)calibration from
        # either surface is immediately reflected in the other's jog_goto
        # calls -- see TransitPanel.set_axis_signs, which mutates in place.
        self.axis_signs = AxisSigns(ra=1.0, dec=1.0)
        # Shared with CalibrationPanel: the checkbox lives in the Transit tab
        # (only useful during an active pass) but CalibrationPanel is what
        # actually reads it to decide whether to apply a correction.
        self.auto_guide_var = tk.BooleanVar(value=False)
        # Shared with CalibrationPanel: "Measure mount lag" there writes it,
        # TransitPanel's start/simulate read it into TrackingConfig.
        self.mount_lag_var = tk.DoubleVar(value=0.0)
        # Owned by TransitPanel's checkbox alone (no writer elsewhere), but
        # held here so it's alongside the other cross-panel tracking state.
        self.feedback_enabled_var = tk.BooleanVar(value=False)
        # Shared with JogWindow (same instance) so its exposure/gain
        # sliders and the Transit tab's stay in sync instead of drifting
        # apart -- see CameraControlVars' docstring.
        self.camera_vars = CameraControlVars.create()
        # Shared with PassesPanel (same instance) so the site entered once
        # is both what passes are searched for AND what the mount is told
        # on connect -- see SiteVars' docstring for the bug this fixes.
        self.site_vars = SiteVars.create()

        self.exposure_panel = ExposurePanel(self.notebook)
        # get_optical_train: so connecting the (mock) camera uses the real
        # plate scale typed into the Exposure calc tab, not a fixed guess.
        # Both device connections live here -- one place to plug everything
        # in before doing anything else, see ConnectionPanel's docstring.
        self.connection_panel = ConnectionPanel(
            self.notebook, self.worker, self.camera_worker, self._on_connection_change,
            get_optical_train=self.exposure_panel.get_optical_train, site_vars=self.site_vars,
        )
        self.passes_panel = PassesPanel(self.notebook, self._on_pass_selected, site_vars=self.site_vars)
        self.transit_panel = TransitPanel(
            self.notebook, self.worker, self.camera_worker, out_dir, self.live_offsets,
            axis_signs=self.axis_signs, auto_guide_var=self.auto_guide_var, camera_vars=self.camera_vars,
            mount_lag_var=self.mount_lag_var, feedback_enabled_var=self.feedback_enabled_var,
        )
        # on_calibration_ready: CalibrationPanel finishes calibration entirely
        # on its own (button click -> internal timers), so it has to tell
        # TransitPanel directly when its checkbox should become enabled --
        # there's no worker event to pump this through.
        self.calibration_panel = CalibrationPanel(
            self.notebook, self.worker, self.camera_worker, self.live_offsets,
            auto_guide_var=self.auto_guide_var,
            on_calibration_ready=lambda: self.transit_panel.set_auto_guide_available(True),
            mount_lag_var=self.mount_lag_var, axis_signs=self.axis_signs,
        )
        # Pure local file I/O, no worker/device -- doesn't need any of the
        # constructor args above, and (unlike every other panel) needs no
        # wiring into _pump_events below.
        self.ser_player_panel = SerPlayerPanel(self.notebook)

        # Plain geometric-shape glyphs (Unicode "Geometric Shapes" block) --
        # unlike pictographic emoji, these render reliably in Tk without
        # depending on a color-emoji font being installed/mapped.
        self.notebook.add(self.connection_panel, text="◆ Connection")
        self.notebook.add(self.passes_panel, text="▲ Passes")
        self.notebook.add(self.calibration_panel, text="⊙ Calibration")
        self.notebook.add(self.exposure_panel, text="▣ Exposure calc")
        self.notebook.add(self.transit_panel, text="◎ Transit")
        self.notebook.add(self.ser_player_panel, text="▶ SER player")

        self._panels = [self.connection_panel]

        # Floating, reachable regardless of the active tab -- created once
        # and hidden/shown (not destroyed) via the button below, see
        # JogWindow.protocol("WM_DELETE_WINDOW", ...).
        self.jog_window = JogWindow(root, self.worker, self.camera_worker, self.axis_signs, camera_vars=self.camera_vars)
        self.jog_window.withdraw()

        # Packed BEFORE the notebook below, even though it's created after
        # -- pack() gives space priority in packing order, not creation
        # order, and the notebook's "expand=True" makes it greedy for any
        # spare height. With the notebook packed first (the original
        # order), a tab whose content grows tall enough (e.g. Passes'
        # multi-line pass detail once a pass is selected) could starve
        # these two of space, in one real case squeezing the log bar down
        # to a sliver -- packing them first reserves their height up
        # front, so the notebook is always the one that gives way.
        jog_button_frame = ttk.Frame(root)
        jog_button_frame.pack(fill="x", side="bottom")
        ttk.Button(jog_button_frame, text="◆ Jog control...", command=self._show_jog_window).pack(anchor="w", padx=10, pady=4)

        log_frame = ttk.LabelFrame(root, text="Log", padding=(6, 2))
        log_frame.pack(fill="x", side="bottom", padx=6, pady=(4, 6))
        # Belt and suspenders on top of the packing-order fix above: pins
        # this frame to a fixed height so it can never be compressed
        # below a usable size regardless of what else is going on.
        log_frame.pack_propagate(False)
        log_frame.configure(height=90)
        # Raw tk.Text -- ttk has no themed text widget, so it needs the
        # palette's colors by hand (see am5/gui/theme.py's docstring).
        self._log_text = tk.Text(
            log_frame, height=4, state="disabled", wrap="word", relief="flat", borderwidth=0,
            background=self.palette.bg_widget, foreground=self.palette.fg_dim,
            insertbackground=self.palette.fg, font=("monospace", 9), padx=6, pady=4,
        )
        self._log_text.pack(fill="both", expand=True)

        self.notebook.pack(fill="both", expand=True, padx=6, pady=(6, 0))

        # (ra_deg, dec_deg) the camera preview treats as "centered", for the
        # idle/manual-jog training view -- see _push_camera_offset.
        self._camera_target_radec: tuple[float, float] | None = None

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(EVENT_POLL_MS, self._pump_events)

    def _show_jog_window(self) -> None:
        self.jog_window.deiconify()
        self.jog_window.lift()
        self.jog_window.focus_force()

    def _on_tab_changed(self, _event: object) -> None:
        if self.notebook.select() == str(self.transit_panel):
            self.transit_panel.focus_preview()

    def _on_connection_change(self, connected: bool) -> None:
        self.jog_window.set_connected(connected)
        self.calibration_panel.set_connected(connected)
        self.transit_panel.set_mount_connected(connected)

    def _on_pass_selected(self, trajectory, window, crossings, site, satellite_name) -> None:
        self.transit_panel.set_trajectory(trajectory, window, crossings, site, satellite_name)
        self.exposure_panel.set_pass(trajectory, window)
        self.calibration_panel.set_trajectory(trajectory)

    def _on_mount_position(self, ra_deg: float, dec_deg: float) -> None:
        """Idle-poll/manual-jog path: the first position sample after a
        reset (connect/goto/park) becomes "centered", so subsequent jogging
        visibly drags the camera's simulated star field/ISS blob off-center
        and back -- training feedback for keeping a target framed by hand.
        During active pass tracking this path is unused (idle polling is
        disabled then; tracking_tick drives the camera instead, see
        _pump_events)."""
        if self._camera_target_radec is None:
            self._camera_target_radec = (ra_deg, dec_deg)
        target_ra_deg, target_dec_deg = self._camera_target_radec
        self.camera_worker.set_sky_context(ra_deg, dec_deg, target_ra_deg, target_dec_deg)

    def _log(self, message: str) -> None:
        self._log_text.configure(state="normal")
        self._log_text.insert("end", f"{datetime.now(timezone.utc).strftime('%H:%M:%S')}  {message}\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _pump_events(self) -> None:
        while True:
            try:
                event = self.worker.events.get_nowait()
            except queue.Empty:
                break
            if event.kind == "calibration_done":
                self.transit_panel.set_axis_signs(AxisSigns(
                    ra=event.payload["ra_sign"], dec=event.payload["dec_sign"],
                    calibrated_pier_side=event.payload.get("pier_side"),
                ))
            if event.kind in LOG_EVENT_KINDS:
                self._log(f"[{event.kind}] {event.payload.get('message', '')}")
            elif event.kind == "goto_result":
                p = event.payload
                target = f"RA={p['target_ra_hours']:.4f}h DEC={p['target_dec_deg']:+.4f}deg" if "target_ra_hours" in p else "?"
                self._log(f"[goto] requested {target} -> :MS# code {p['code']}: {p['meaning']}")
            elif event.kind == "goto_arrived":
                if "ra_hours" in event.payload:
                    self._log(f"[goto] arrived at RA={event.payload['ra_hours']:.4f}h DEC={event.payload['dec_deg']:+.4f}deg")
            if event.kind in CAMERA_REFERENCE_RESET_EVENTS:
                self._camera_target_radec = None
            elif event.kind == "position":
                self._on_mount_position(event.payload["ra_hours"] * 15.0, event.payload["dec_deg"])
                # NOT auto-correcting axis_signs.dec from idle-poll :Gm#
                # reads anymore -- tried and reverted along with the
                # jog_goto/run_tracking_loop call sites, see AxisSigns'
                # docstring in tracker.py for why. Recalibrate by hand
                # after any deliberate re-point instead.
            elif event.kind == "tracking_tick" and event.payload["actual_ra_deg"] != "":
                self.camera_worker.set_sky_context(
                    event.payload["actual_ra_deg"], event.payload["actual_dec_deg"],
                    event.payload["target_ra_deg"], event.payload["target_dec_deg"],
                )
            for panel in self._panels:
                panel.handle_event(event)
            self.transit_panel.handle_mount_event(event)
            self.calibration_panel.handle_mount_event(event)
            self.jog_window.handle_mount_event(event)

        while True:
            try:
                event = self.camera_worker.events.get_nowait()
            except queue.Empty:
                break
            if event.kind in LOG_EVENT_KINDS:
                self._log(f"[camera:{event.kind}] {event.payload.get('message', '')}")
            self.connection_panel.handle_camera_event(event)
            self.transit_panel.handle_camera_event(event)
            self.calibration_panel.handle_camera_event(event)
            self.jog_window.handle_camera_event(event)

        self.root.after(EVENT_POLL_MS, self._pump_events)

    def _on_close(self) -> None:
        self.worker.shutdown()
        self.camera_worker.shutdown()
        self.root.destroy()


def run(out_dir: Path | None = None) -> None:
    root = tk.Tk()
    App(root, out_dir or Path("logs"))
    root.mainloop()
