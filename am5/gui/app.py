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

from am5.gui.finder_window import FinderWindow
from am5.gui.jog_window import JogWindow
from am5.gui.ser_player_window import SerPlayerWindow
from am5.gui.panels import (
    AlignmentPanel,
    CameraControlVars,
    ConnectionPanel,
    ExposurePanel,
    CalibrationPanel,
    FinderCameraPanel,
    PassesPanel,
    SiteVars,
    TransitPanel,
)
from am5.gui.theme import apply_dark_theme
from am5.gui.worker import MountWorker
from am5.tracker import AxisSigns, LiveOffsets
from camera.finder import FinderState
from camera.worker import CameraEvent, CameraWorker, pgm_to_array

# Events after which the operator has just confirmed/reset "on target" --
# the idle-mode camera reference re-acquires from the next position sample
# rather than carrying over a stale offset from before the move.
CAMERA_REFERENCE_RESET_EVENTS = {"connected", "goto_arrived", "parked"}

EVENT_POLL_MS = 100
LOG_EVENT_KINDS = {"log", "connect_error", "tracking_error"}


def _drain_coalescing_preview_frames(event_queue: "queue.Queue[CameraEvent]") -> list[CameraEvent]:
    """Drains every currently-queued event from a CameraWorker's events
    queue, but collapses a run of CONSECUTIVE "preview_frame" events down
    to just the last one in that run -- if _pump_events ever falls behind
    for any reason (a slow one-off redraw elsewhere on the Tk thread,
    e.g. TransitPanel's own sky-map redraw when Simulate is clicked, or
    simply the machine being briefly busy), the camera worker's own read
    loop keeps producing preview_frame events in the background
    regardless of whether anyone's drawing them. Rendering each stale one
    in turn (a real tk.PhotoImage + canvas update per panel watching it
    -- not cheap, confirmed via profiling: TransitPanel's own preview
    update alone measured as the single largest cost in a normal
    _pump_events call) only makes an existing backlog worse, and nothing
    before the last frame in a consecutive run was ever actually visible
    anyway. Only collapses a run that's actually consecutive (not any two
    preview_frame events anywhere in the batch) so every OTHER event kind
    stays exactly where it was in the original queue order -- a
    preview_frame that arrived after a "stats"/"log"/etc. event must not
    end up reordered to look like it arrived before it."""
    events: list[CameraEvent] = []
    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            break
        if event.kind == "preview_frame" and events and events[-1].kind == "preview_frame":
            events[-1] = event  # supersedes the immediately-preceding stale one
        else:
            events.append(event)
    return events


class App:
    def __init__(self, root: tk.Tk, out_dir: Path):
        self.root = root
        self.worker = MountWorker()
        self.camera_worker = CameraWorker()
        # Second camera (finder scope) -- entirely optional, stays
        # disconnected/disabled if not plugged in. Owns a separate
        # CameraWorker so the finder and main cameras run independently.
        self.finder_worker = CameraWorker()
        self.finder_state = FinderState()

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
        # Shared the same way as mount_lag_var -- "Measure mount lag" also
        # derives this (steady_rate/lag_s) alongside it. 0.0 means "not
        # measured", which TrackingConfig.max_accel_deg_s2=None (the old
        # flat mount_lag_s behavior) treats the same as "disabled".
        self.mount_max_accel_var = tk.DoubleVar(value=0.0)
        # Owned by TransitPanel's checkbox alone (no writer elsewhere), but
        # held here so it's alongside the other cross-panel tracking state.
        self.feedback_enabled_var = tk.BooleanVar(value=False)
        # Shared with JogWindow (same instance) so its exposure/gain
        # sliders and the Transit tab's stay in sync instead of drifting
        # apart -- see CameraControlVars' docstring.
        self.camera_vars = CameraControlVars.create()
        # Same fix, for the finder camera's own two independent slider
        # widgets (FinderCameraPanel's tab and the floating FinderWindow) --
        # reported as a real bug: changing one didn't move the other, and
        # each silently kept driving whatever it last showed.
        self.finder_camera_vars = CameraControlVars.create(
            FinderCameraPanel.FINDER_DEFAULT_EXPOSURE_US, FinderCameraPanel.FINDER_DEFAULT_GAIN,
        )
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
            finder_worker=self.finder_worker, finder_state=self.finder_state,
        )
        self.passes_panel = PassesPanel(self.notebook, self._on_pass_selected, site_vars=self.site_vars)
        # on_tracking_trajectory_changed: CalibrationPanel's own auto-guide
        # correction needs the trajectory ACTUALLY being tracked right now
        # (possibly Simulate's own time-shifted copy), not just whichever
        # pass is selected -- see TransitPanel._active_trajectory's own
        # comment. A lambda, not a direct method reference, since
        # self.calibration_panel doesn't exist yet at this point in
        # construction (same pattern as on_calibration_ready below,
        # resolved at call time instead).
        self.transit_panel = TransitPanel(
            self.notebook, self.worker, self.camera_worker, out_dir, self.live_offsets,
            axis_signs=self.axis_signs, auto_guide_var=self.auto_guide_var, camera_vars=self.camera_vars,
            mount_lag_var=self.mount_lag_var, mount_max_accel_var=self.mount_max_accel_var,
            feedback_enabled_var=self.feedback_enabled_var,
            finder_state=self.finder_state,
            on_tracking_trajectory_changed=lambda t: self.calibration_panel.set_active_trajectory(t),
        )
        # on_calibration_ready/on_finder_calibration_ready: CalibrationPanel
        # finishes each calibration entirely on its own (button click ->
        # internal timers/FFT correlation), so it has to tell TransitPanel
        # directly when each checkbox should become enabled -- there's no
        # worker event to pump either through.
        self.calibration_panel = CalibrationPanel(
            self.notebook, self.worker, self.camera_worker, self.live_offsets,
            auto_guide_var=self.auto_guide_var,
            on_calibration_ready=lambda: self.transit_panel.set_auto_guide_available(True),
            mount_lag_var=self.mount_lag_var, mount_max_accel_var=self.mount_max_accel_var,
            axis_signs=self.axis_signs, finder_state=self.finder_state,
            on_finder_calibration_ready=lambda: self.transit_panel.set_finder_correction_available(True),
            camera_vars=self.camera_vars, finder_camera_vars=self.finder_camera_vars,
        )
        self.alignment_panel = AlignmentPanel(
            self.notebook, self.worker, self.axis_signs, self.site_vars, finder_state=self.finder_state,
            camera_vars=self.camera_vars, finder_camera_vars=self.finder_camera_vars, out_dir=out_dir,
        )

        # Finder camera panel -- optional wide-field second camera for ISS
        # acquisition.  Created unconditionally but greyed out until a
        # finder camera is actually connected via its own Connect button.
        # Field calibration itself now lives in CalibrationPanel (see its
        # on_finder_calibration_ready above), alongside the main camera's
        # own calibration -- this panel keeps the live preview/exposure/
        # tracking-offset controls only.
        self.finder_panel = FinderCameraPanel(
            self.notebook, self.finder_worker, self.finder_state,
            live_offsets=self.live_offsets,
            camera_vars=self.finder_camera_vars,
        )

        # Plain geometric-shape glyphs (Unicode "Geometric Shapes" block) --
        # unlike pictographic emoji, these render reliably in Tk without
        # depending on a color-emoji font being installed/mapped.
        #
        self.notebook.add(self.connection_panel, text="◆ Connection")
        self.notebook.add(self.passes_panel, text="▲ Passes")
        self.notebook.add(self.alignment_panel, text="✦ Alignment")
        self.notebook.add(self.calibration_panel, text="⊙ Calibration")
        self.notebook.add(self.exposure_panel, text="▣ Exposure calc")
        self.notebook.add(self.finder_panel, text="🔭 Finder")
        self.notebook.add(self.transit_panel, text="◎ Transit")

        self._panels = [self.connection_panel]

        # Floating, reachable regardless of the active tab -- created once
        # and hidden/shown (not destroyed) via the button below, see
        # JogWindow.protocol("WM_DELETE_WINDOW", ...).
        self.jog_window = JogWindow(root, self.worker, self.camera_worker, self.axis_signs, camera_vars=self.camera_vars)
        self.jog_window.withdraw()

        self.finder_window = FinderWindow(
            root,
            self.finder_worker,
            self.finder_state,
            on_sync=lambda ra, dec: self.worker.sync(ra / 15.0, dec),
            camera_vars=self.finder_camera_vars,
        )
        self.finder_window.withdraw()

        # SER player is a standalone review tool (plays back a recorded
        # file, no live device/session state involved) -- a separate
        # floating window rather than a Notebook tab, same reasoning as
        # JogWindow/FinderWindow: a plain tab's position in a wrapping
        # multi-row tab strip isn't stable (with enough other tabs it
        # could land at the start of a wrapped second row instead of
        # staying visually separate from the workflow tabs).
        self.ser_player_window = SerPlayerWindow(root)
        self.ser_player_window.withdraw()

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
        ttk.Button(jog_button_frame, text="◆ Jog control...", command=self._show_jog_window).pack(
            side="left", padx=10, pady=4,
        )
        ttk.Button(jog_button_frame, text="🔭 Finder...", command=self._show_finder_window).pack(
            side="left", pady=4,
        )
        ttk.Button(jog_button_frame, text="▶ SER player...", command=self._show_ser_player_window).pack(
            side="left", pady=4,
        )

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

    def _show_finder_window(self) -> None:
        self.finder_window.deiconify()
        self.finder_window.lift()
        self.finder_window.focus_force()

    def _show_ser_player_window(self) -> None:
        self.ser_player_window.deiconify()
        self.ser_player_window.lift()
        self.ser_player_window.focus_force()

    def _on_tab_changed(self, _event: object) -> None:
        if self.notebook.select() == str(self.transit_panel):
            self.transit_panel.focus_preview()

    def _on_connection_change(self, connected: bool) -> None:
        self.jog_window.set_connected(connected)
        self.calibration_panel.set_connected(connected)
        self.alignment_panel.set_connected(connected)
        self.transit_panel.set_mount_connected(connected)

    def _on_pass_selected(self, trajectory, window, crossings, site, satellite_name) -> None:
        self.transit_panel.set_trajectory(trajectory, window, crossings, site, satellite_name)
        self.exposure_panel.set_pass(trajectory, window)
        # CalibrationPanel no longer gets the trajectory here -- selecting
        # a pass can happen hours before it starts, and its own auto-guide
        # correction needs the trajectory ACTUALLY being tracked right
        # now, not just whichever pass is selected (see TransitPanel.
        # _active_trajectory's own comment). Wired instead from
        # TransitPanel's own Start/Simulate/stop handling, via
        # on_tracking_trajectory_changed in this file's own constructor
        # wiring above.
        self.alignment_panel.set_trajectory(trajectory, window, satellite_name)

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
        # Finder shares the same mount/boresight as the main camera (it's
        # bolted to the same tube) -- same sky context, just rendered at
        # the finder's own wider plate scale/FOV (see MockAsiCamera's own
        # _plate_scale, set from ConnectionPanel.FINDER_PLATE_SCALE at
        # connect time).
        self.finder_worker.set_sky_context(ra_deg, dec_deg, target_ra_deg, target_dec_deg)

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
                self.finder_worker.set_sky_context(
                    event.payload["actual_ra_deg"], event.payload["actual_dec_deg"],
                    event.payload["target_ra_deg"], event.payload["target_dec_deg"],
                )
            for panel in self._panels:
                panel.handle_event(event)
            self.transit_panel.handle_mount_event(event)
            self.calibration_panel.handle_mount_event(event)
            self.alignment_panel.handle_mount_event(event)
            self.jog_window.handle_mount_event(event)
            self.finder_window.handle_mount_event(event)

        for event in _drain_coalescing_preview_frames(self.camera_worker.events):
            if event.kind in LOG_EVENT_KINDS:
                self._log(f"[camera:{event.kind}] {event.payload.get('message', '')}")
            if event.kind == "preview_frame":
                # Feeds FinderState.last_main_frame so "Calibrate fields"
                # (FinderCameraPanel._on_calibrate) can correlate the finder
                # against a REAL main-camera frame instead of silently
                # falling back to correlating the finder against itself.
                self.finder_state.update_main_frame(pgm_to_array(event.payload["pgm"]))
            self.connection_panel.handle_camera_event(event)
            self.transit_panel.handle_camera_event(event)
            self.calibration_panel.handle_camera_event(event)
            self.jog_window.handle_camera_event(event)

        for event in _drain_coalescing_preview_frames(self.finder_worker.events):
            if event.kind in LOG_EVENT_KINDS:
                self._log(f"[finder:{event.kind}] {event.payload.get('message', '')}")
            if event.kind == "disconnected":
                # Clears any stale ISS-blob detection so a dropped finder
                # connection can't keep silently driving cross-track
                # corrections mid-pass (see FinderState.reset_blob).
                self.finder_state.reset_blob()
            self.connection_panel.handle_finder_camera_event(event)
            self.calibration_panel.handle_finder_camera_event(event)
            self.finder_panel.handle_camera_event(event)
            self.finder_window.handle_camera_event(event)

        self.root.after(EVENT_POLL_MS, self._pump_events)

    def _on_close(self) -> None:
        self.worker.shutdown()
        self.camera_worker.shutdown()
        self.finder_worker.shutdown()
        self.root.destroy()


def run(out_dir: Path | None = None) -> None:
    root = tk.Tk()
    App(root, out_dir or Path("logs"))
    root.mainloop()
