"""Floating jog-control window: RA/DEC readout, arrow-key/button jogging,
park/unpark, tracking/altitude-limit toggles, a "GOTO a named star" list,
an emergency stop, and camera exposure/gain -- usable regardless of which
Notebook tab is currently selected (see am5/gui/app.py, which owns the one
instance and just shows/hides it rather than destroying it on close).

This used to be split across a separate "Manual control" tab and this
window; folded into one here since almost everything in that tab already
had an equivalent (or a better, pier-preserving one) in this window, and
having both meant two places to look for the same controls.

Deliberately thin: talks to the mount only through MountWorker's existing
thread-safe API -- no new device-touching code here. Every control whose
worker-side handler is blocked_while_parked (jog buttons, rate entry,
tracking checkbox, GOTO->) is greyed out while parked/disconnected -- see
_refresh_widget_states/set_connected. This isn't just cosmetic: several of
these disable themselves on click and wait for a reply event to re-enable
(GOTO -> waits for jog_goto_result, etc.) -- clicking them while parked
used to leave the button disabled forever, since the blocked handler logs
a warning and returns without ever emitting that event. Sync/stop/altitude
limits aren't blocked_while_parked on the worker side, so they only grey
out when disconnected. EMERGENCY STOP is never gated by anything.
Exposure/gain duplicate a small slice of TransitPanel's camera controls
(same CameraWorker, and the same shared CameraControlVars -- see
am5/gui/panels.py -- so the two sliders stay in sync instead of drifting
apart) so exposure/gain can be nudged without switching to the Transit tab
while working the ISS into frame -- the ROI/preview/recording controls
stay Transit-only, they don't fit this window's "quick control, any tab"
purpose.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

from am5.gui.panels import MAX_EXPOSURE_SLIDER_US, CameraControlVars
from am5.gui.theme import PALETTE
from am5.gui.worker import MountWorker, WorkerEvent
from am5.named_stars import NAMED_STARS, NAMED_STARS_BY_NAME
from am5.tracker import AxisSigns
from camera.worker import CameraEvent, CameraWorker


class JogWindow(tk.Toplevel):
    _KEY_DIRECTIONS = {"Up": "n", "Down": "s", "Left": "w", "Right": "e"}

    def __init__(
        self, parent: tk.Misc, mount_worker: MountWorker, camera_worker: CameraWorker, axis_signs: AxisSigns,
        camera_vars: CameraControlVars | None = None,
    ):
        super().__init__(parent)
        self.title("Mount jog control")
        # Toplevel is a raw tk widget -- apply_dark_theme's ttk.Style
        # doesn't reach it (see am5/gui/theme.py's docstring).
        self.configure(background=PALETTE.bg)
        self._mount_worker = mount_worker
        self._camera_worker = camera_worker
        self._camera_interactive_widgets: list[tk.Widget] = []
        self._axis_signs = axis_signs  # shared with App/TransitPanel -- always current, see app.py
        # Shared with TransitPanel (same instance, owned by App) when
        # passed -- see CameraControlVars' docstring.
        self._camera_vars = camera_vars if camera_vars is not None else CameraControlVars.create()
        self._connected = False
        self._parked = False
        # Gated by "connected" alone (stop/sync/alt-limits aren't
        # blocked_while_parked on the worker side).
        self._interactive_widgets: list[tk.Widget] = []
        # Gated by "connected AND NOT parked" -- their worker-side handlers
        # are blocked_while_parked, and several of these disable themselves
        # on click and wait for a reply event to re-enable -- a real
        # incident: clicking "GOTO ->" while parked left that button
        # disabled forever, since jog_goto's handler just logs a warning
        # and returns without ever emitting jog_goto_result.
        self._motion_widgets: list[tk.Widget] = []

        ttk.Label(self, text="MOUNT JOG CONTROL", font=("", 10, "bold"), foreground=PALETTE.accent).pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self._position_var = tk.StringVar(value="RA: --  DEC: --")
        ttk.Label(self, textvariable=self._position_var, font=("", 13)).pack(anchor="w", padx=10, pady=(2, 8))

        columns = ttk.Frame(self)
        columns.pack(fill="x", padx=10)
        left = ttk.Frame(columns)
        left.pack(side="left", anchor="n")
        right = ttk.Frame(columns)
        right.pack(side="left", anchor="n", padx=(20, 0))

        rate_row = ttk.Frame(left)
        rate_row.pack(anchor="w")
        ttk.Label(rate_row, text="rate (x sidereal)").pack(side="left")
        self._rate_var = tk.StringVar(value="60")
        self._rate_entry = ttk.Entry(rate_row, textvariable=self._rate_var, width=8)
        self._rate_entry.pack(side="left", padx=(4, 0))
        self._motion_widgets.append(self._rate_entry)

        jog_frame = ttk.Frame(left)
        jog_frame.pack(pady=12)
        # Keyed by direction so keyboard-triggered jogs (see
        # _on_jog_key_press/_on_jog_key_release) can visually press the
        # matching button -- a mouse click already shows this for free
        # (Tk's own default Button behavior), but a keypress touches no
        # widget at all otherwise, giving no feedback that it registered.
        self._jog_buttons: dict[str, ttk.Button] = {
            "n": self._make_jog_button(jog_frame, "▲", "n", row=0, col=1),
            "w": self._make_jog_button(jog_frame, "◀", "w", row=1, col=0),
        }
        stop_button = ttk.Button(jog_frame, text="■", width=4, command=self._mount_worker.stop_all)
        stop_button.grid(row=1, column=1, padx=2, pady=2)
        self._interactive_widgets.append(stop_button)
        self._jog_buttons["e"] = self._make_jog_button(jog_frame, "▶", "e", row=1, col=2)
        self._jog_buttons["s"] = self._make_jog_button(jog_frame, "▼", "s", row=2, col=1)

        ttk.Label(left, text="Arrow keys jog too, anywhere in\nthis window (click it first for\nkeyboard focus)",
                  foreground=PALETTE.fg_dim, justify="left").pack(anchor="w")

        park_frame = ttk.Frame(right)
        park_frame.pack(fill="x")
        self._park_button = ttk.Button(park_frame, text="Park (:hC# home)", command=self._on_park_click)
        self._park_button.pack(side="left")
        self._park_native_button = ttk.Button(park_frame, text="Park (native :hP#)", command=self._on_park_native_click)
        self._park_native_button.pack(side="left", padx=(4, 0))
        self._unpark_button = ttk.Button(park_frame, text="Unpark", command=self._on_unpark_click, state="disabled")
        self._unpark_button.pack(side="left", padx=(4, 0))
        self._park_status_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self._park_status_var, foreground=PALETTE.accent_warn).pack(anchor="w")
        ttk.Label(right, text="(native :hP# is untested — INDI's own AM5 driver avoids it; try both and compare)",
                  foreground=PALETTE.fg_dim, wraplength=320, justify="left").pack(anchor="w")

        self._tracking_var = tk.BooleanVar(value=False)
        self._tracking_check = ttk.Checkbutton(
            right, text="Sidereal tracking", variable=self._tracking_var,
            command=lambda: self._mount_worker.set_tracking(self._tracking_var.get()),
        )
        self._tracking_check.pack(anchor="w", pady=(4, 0))
        self._motion_widgets.append(self._tracking_check)

        self._alt_limits_var = tk.BooleanVar(value=True)
        alt_limits_check = ttk.Checkbutton(
            right, text="Altitude limits enabled", variable=self._alt_limits_var, command=self._on_alt_limits_toggle,
        )
        alt_limits_check.pack(anchor="w")
        self._interactive_widgets.append(alt_limits_check)
        self._alt_limits_warning = ttk.Label(right, text="", foreground=PALETTE.accent_bad)
        self._alt_limits_warning.pack(anchor="w")

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)

        goto_frame = ttk.Frame(right)
        goto_frame.pack(fill="x")
        ttk.Label(goto_frame, text="GOTO / Sync a star", font=("", 9, "bold")).pack(anchor="w")
        ttk.Label(
            goto_frame,
            text="GOTO: jog-based, preserves pier side. Sync: tells the mount it's\n"
                 "already centered on the selected star, no motion -- use after\n"
                 "manually centering one to correct accumulated pointing error.",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(anchor="w")
        row2 = ttk.Frame(goto_frame)
        row2.pack(anchor="w", pady=(6, 0))
        star_names = [s.name for s in sorted(NAMED_STARS, key=lambda s: s.magnitude)]  # brightest first
        self._star_var = tk.StringVar(value=star_names[0] if star_names else "")
        ttk.Combobox(row2, textvariable=self._star_var, values=star_names, state="readonly", width=18).pack(side="left")
        self._goto_star_button = ttk.Button(row2, text="GOTO →", command=self._on_goto_star_click)
        self._goto_star_button.pack(side="left", padx=(4, 0))
        self._motion_widgets.append(self._goto_star_button)
        self._sync_star_button = ttk.Button(row2, text="Sync", command=self._on_sync_star_click)
        self._sync_star_button.pack(side="left", padx=(4, 0))
        self._interactive_widgets.append(self._sync_star_button)  # not gated by parked -- sync never moves the mount
        self._goto_status_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self._goto_status_var, foreground=PALETTE.accent_ok).pack(anchor="w", pady=(4, 0))

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=10, pady=10)

        camera_frame = ttk.LabelFrame(self, text="Camera exposure / gain", padding=8)
        camera_frame.pack(fill="x", padx=10, pady=(0, 10))
        camera_frame.columnconfigure(1, weight=1)

        ttk.Label(camera_frame, text="Exposure").grid(row=0, column=0, sticky="w")
        self._exposure_scale = ttk.Scale(
            camera_frame, from_=math.log10(32), to=math.log10(MAX_EXPOSURE_SLIDER_US),
            variable=self._camera_vars.exposure_log, state="disabled",
        )
        self._exposure_scale.grid(row=0, column=1, sticky="we", padx=(8, 8))
        self._exposure_scale.bind("<ButtonRelease-1>", self._on_exposure_release)
        self._camera_interactive_widgets.append(self._exposure_scale)
        ttk.Label(camera_frame, textvariable=self._camera_vars.exposure_value, width=10).grid(row=0, column=2, sticky="w")

        ttk.Label(camera_frame, text="Gain").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._gain_scale = ttk.Scale(
            camera_frame, from_=0, to=570, variable=self._camera_vars.gain, state="disabled",
        )
        self._gain_scale.grid(row=1, column=1, sticky="we", padx=(8, 8), pady=(6, 0))
        self._gain_scale.bind("<ButtonRelease-1>", self._on_gain_release)
        self._camera_interactive_widgets.append(self._gain_scale)
        ttk.Label(camera_frame, textvariable=self._camera_vars.gain_value, width=10).grid(row=1, column=2, sticky="w", pady=(6, 0))

        # Deliberately not gated by connected/parked: stays live regardless
        # -- the one control that must never be blocked behind anything.
        estop = tk.Button(
            self, text="EMERGENCY STOP", command=self._mount_worker.emergency_stop,
            bg="#c00", fg="white", font=("", 14, "bold"), height=2,
        )
        estop.pack(fill="x", padx=10, pady=(0, 10))

        # Bound on every widget in the window, not just the Toplevel itself:
        # a plain self.bind() only fires when nothing more specific has
        # already consumed the key, but ttk.Entry/ttk.Scale/ttk.Combobox
        # all have their own built-in Left/Right/Up/Down bindings (cursor
        # move, value nudge, ...) that run first and swallow the event
        # whenever one of them happens to have focus -- e.g. after typing a
        # rate or dragging a slider. Binding directly on each widget (and
        # returning "break") makes arrow keys always jog the mount while
        # this window has focus, regardless of which control was clicked
        # last.
        self._bind_jog_keys(self)

        # Closing the window (X button) just hides it -- App keeps the one
        # instance alive so events keep flowing and it can be reopened
        # without losing state.
        self.protocol("WM_DELETE_WINDOW", self.withdraw)

        self.set_connected(False)

    def _bind_jog_keys(self, widget: tk.Misc) -> None:
        for keyname, direction in self._KEY_DIRECTIONS.items():
            widget.bind(f"<{keyname}>", lambda e, d=direction: self._on_jog_key_press(d, e))
            widget.bind(f"<KeyRelease-{keyname}>", lambda e, d=direction: self._on_jog_key_release(d, e))
        for child in widget.winfo_children():
            self._bind_jog_keys(child)

    def _on_jog_key_press(self, direction: str, _event: tk.Event) -> str:
        self._mount_worker.jog_start(direction, self._current_rate())
        button = self._jog_buttons.get(direction)
        if button is not None:
            button.state(["pressed"])
        return "break"  # pre-empts the focused widget's own arrow-key handling (see _bind_jog_keys)

    def _on_jog_key_release(self, direction: str, _event: tk.Event) -> str:
        self._mount_worker.jog_stop(direction)
        button = self._jog_buttons.get(direction)
        if button is not None:
            button.state(["!pressed"])
        return "break"

    def _make_jog_button(self, parent: tk.Misc, label: str, direction: str, row: int, col: int) -> ttk.Button:
        button = ttk.Button(parent, text=label, width=4, style="Jog.TButton")
        button.grid(row=row, column=col, padx=2, pady=2)
        button.bind("<ButtonPress-1>", lambda _e: self._mount_worker.jog_start(direction, self._current_rate()))
        button.bind("<ButtonRelease-1>", lambda _e: self._mount_worker.jog_stop(direction))
        self._motion_widgets.append(button)
        return button

    def _current_rate(self) -> float:
        try:
            return float(self._rate_var.get())
        except ValueError:
            return 0.0

    def _on_park_click(self) -> None:
        self._park_status_var.set("Parking (:hC#)...")
        self._mount_worker.park()

    def _on_park_native_click(self) -> None:
        self._park_status_var.set("Parking (native :hP#)...")
        self._mount_worker.park_native()

    def _on_unpark_click(self) -> None:
        self._mount_worker.unpark()

    def _on_alt_limits_toggle(self) -> None:
        enabled = self._alt_limits_var.get()
        self._mount_worker.set_altitude_limits(enabled)
        self._alt_limits_warning.configure(text="" if enabled else "WARNING: limits disabled — remember to re-enable")

    def _refresh_widget_states(self) -> None:
        for widget in self._interactive_widgets:
            widget.configure(state="normal" if self._connected else "disabled")
        for widget in self._motion_widgets:
            widget.configure(state="normal" if (self._connected and not self._parked) else "disabled")
        can_park = self._connected and not self._parked
        self._park_button.configure(state="normal" if can_park else "disabled")
        self._park_native_button.configure(state="normal" if can_park else "disabled")
        self._unpark_button.configure(state="normal" if (self._connected and self._parked) else "disabled")

    def set_connected(self, connected: bool) -> None:
        self._connected = connected
        if not connected:
            self._parked = False
            self._position_var.set("RA: --  DEC: --")
            self._park_status_var.set("")
        self._refresh_widget_states()

    def _on_goto_star_click(self) -> None:
        star = NAMED_STARS_BY_NAME.get(self._star_var.get())
        if star is None:
            return
        self._goto_star_button.configure(state="disabled")
        self._goto_status_var.set(f"GOTO {star.name} (RA={star.ra_hours:.3f}h DEC={star.dec_deg:+.2f} deg)...")
        self._mount_worker.jog_goto(star.ra_hours, star.dec_deg, self._axis_signs)

    def _on_sync_star_click(self) -> None:
        star = NAMED_STARS_BY_NAME.get(self._star_var.get())
        if star is None:
            return
        self._sync_star_button.configure(state="disabled")
        self._goto_status_var.set(f"Syncing to {star.name} (RA={star.ra_hours:.3f}h DEC={star.dec_deg:+.2f} deg)...")
        self._mount_worker.sync(star.ra_hours, star.dec_deg)

    def handle_mount_event(self, event: WorkerEvent) -> None:
        if event.kind == "position":
            self._position_var.set(f"RA: {event.payload['ra_hours']:.4f}h  DEC: {event.payload['dec_deg']:+.4f} deg")
        elif event.kind == "tracking_tick":
            actual_ra_deg = event.payload["actual_ra_deg"]
            if actual_ra_deg != "":  # only populated every error_log_every ticks, see am5/tracker.py
                self._position_var.set(f"RA: {actual_ra_deg / 15.0:.4f}h  DEC: {event.payload['actual_dec_deg']:+.4f} deg")
        elif event.kind == "jog_goto_result":
            self._goto_status_var.set("Arrived" if event.payload.get("arrived") else "Did not arrive -- check the log below")
            self._refresh_widget_states()  # not a flat "normal" -- respects a park that landed mid-GOTO
        elif event.kind == "sync_result":
            self._goto_status_var.set(event.payload["message"] if event.payload["ok"] else f"Sync failed: {event.payload['message']}")
            self._refresh_widget_states()
        elif event.kind == "parked":
            self._parked = True
            method = event.payload.get("method", "home")
            reply = event.payload.get("reply")
            detail = f" (reply: {reply})" if reply is not None else ""
            self._park_status_var.set(f"Parked via {method}{detail} — unpark before moving the mount again")
            self._refresh_widget_states()
        elif event.kind == "unparked":
            self._parked = False
            self._park_status_var.set("")
            self._refresh_widget_states()

    # -- exposure / gain sliders ---------------------------------------------
    # Live label updates are handled by CameraControlVars' own traces (see
    # am5/gui/panels.py) -- only the on-release commit to the camera is
    # per-widget here, same reasoning as TransitPanel's copy of this.

    def _on_exposure_release(self, _event: tk.Event) -> None:
        self._camera_worker.set_exposure_us(round(10 ** self._camera_vars.exposure_log.get()))

    def _on_gain_release(self, _event: tk.Event) -> None:
        self._camera_worker.set_gain(round(self._camera_vars.gain.get()))

    def _configure_control_bounds(self, controls: dict) -> None:
        exposure = controls.get("Exposure")
        if exposure:
            lo = max(32, exposure["MinValue"])
            hi = min(MAX_EXPOSURE_SLIDER_US, max(lo + 1, exposure["MaxValue"]))
            self._exposure_scale.configure(from_=math.log10(lo), to=math.log10(hi))
            default = min(hi, max(lo, exposure.get("DefaultValue", lo)))
            self._camera_vars.exposure_log.set(math.log10(default))
        gain = controls.get("Gain")
        if gain:
            self._gain_scale.configure(from_=gain["MinValue"], to=gain["MaxValue"])
            default = gain.get("DefaultValue", gain["MinValue"])
            self._camera_vars.gain.set(default)

    def handle_camera_event(self, event: CameraEvent) -> None:
        if event.kind == "connected":
            for widget in self._camera_interactive_widgets:
                widget.configure(state="normal")
            self._configure_control_bounds(event.payload.get("controls", {}))
        elif event.kind == "disconnected":
            for widget in self._camera_interactive_widgets:
                widget.configure(state="disabled")
