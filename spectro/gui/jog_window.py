"""Floating jog-control window, usable regardless of which Notebook tab
is currently selected -- same pattern as am5/gui/jog_window.py (the ISS
tracker's own JogWindow): a single instance, created once and shown/
hidden (not destroyed) so it keeps receiving events and never loses
state, talking to the mount only through MountWorker's existing thread-
safe API.

Deliberately a smaller window than the ISS tracker's version: no camera
exposure/gain section (this app's CameraWorker only opens the device --
see gui/panels.py's module docstring -- and each AcquisitionPanel tab
already has its own exposure/gain sliders for the synthetic frames it
generates, so a second, real-camera-only control here would be redundant
until frame acquisition itself is real) and no "GOTO a named star" list
(that version's GOTO uses a closed-loop jog_goto requiring a calibrated
AxisSigns, which nothing in this app calibrates; each AcquisitionPanel tab
already has a more directly useful GOTO button wired to the actual
SIMBAD-resolved reference/target star instead of a fixed named-star list,
see panels.py's AcquisitionPanel._on_goto).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from am5.gui.theme import PALETTE
from am5.gui.worker import MountWorker, WorkerEvent


class JogWindow(tk.Toplevel):
    _KEY_DIRECTIONS = {"Up": "n", "Down": "s", "Left": "w", "Right": "e"}

    def __init__(self, parent: tk.Misc, mount_worker: MountWorker):
        super().__init__(parent)
        self.title("Mount jog control")
        # Toplevel is a raw tk widget -- apply_dark_theme's ttk.Style
        # doesn't reach it (see am5/gui/theme.py's docstring).
        self.configure(background=PALETTE.bg)
        self._mount_worker = mount_worker
        self._connected = False
        self._parked = False
        # Gated by "connected" alone (stop/alt-limits aren't
        # blocked_while_parked on the worker side).
        self._interactive_widgets: list[tk.Widget] = []
        # Gated by "connected AND NOT parked" -- their worker-side
        # handlers ARE blocked_while_parked.
        self._motion_widgets: list[tk.Widget] = []

        ttk.Label(self, text="MOUNT JOG CONTROL", font=("", 10, "bold"), foreground=PALETTE.accent).pack(
            anchor="w", padx=10, pady=(10, 0),
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
        self._jog_buttons: dict[str, ttk.Button] = {
            "n": self._make_jog_button(jog_frame, "▲", "n", row=0, col=1),
            "w": self._make_jog_button(jog_frame, "◀", "w", row=1, col=0),
        }
        stop_button = ttk.Button(jog_frame, text="■", width=4, command=self._mount_worker.stop_all)
        stop_button.grid(row=1, column=1, padx=2, pady=2)
        self._interactive_widgets.append(stop_button)
        self._jog_buttons["e"] = self._make_jog_button(jog_frame, "▶", "e", row=1, col=2)
        self._jog_buttons["s"] = self._make_jog_button(jog_frame, "▼", "s", row=2, col=1)

        ttk.Label(
            left, text="Arrow keys jog too, anywhere in\nthis window (click it first for\nkeyboard focus)",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(anchor="w")

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

        # Deliberately not gated by connected/parked: stays live regardless
        # -- the one control that must never be blocked behind anything.
        estop = tk.Button(
            self, text="EMERGENCY STOP", command=self._mount_worker.emergency_stop,
            bg="#c00", fg="white", font=("", 14, "bold"), height=2,
        )
        estop.pack(fill="x", padx=10, pady=10)

        # Bound on every widget in the window, not just the Toplevel itself
        # -- ttk.Entry/ttk.Scale/ttk.Combobox all have their own built-in
        # Left/Right/Up/Down bindings that would otherwise swallow the
        # event whenever one of them has focus. See am5/gui/jog_window.py's
        # own copy of this same fix for the incident it avoids.
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
        self._alt_limits_warning.configure(text="" if enabled else "WARNING: limits disabled -- remember to re-enable")

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

    def handle_mount_event(self, event: WorkerEvent) -> None:
        if event.kind == "position":
            self._position_var.set(f"RA: {event.payload['ra_hours']:.4f}h  DEC: {event.payload['dec_deg']:+.4f} deg")
        elif event.kind == "parked":
            self._parked = True
            method = event.payload.get("method", "home")
            reply = event.payload.get("reply")
            detail = f" (reply: {reply})" if reply is not None else ""
            self._park_status_var.set(f"Parked via {method}{detail} -- unpark before moving the mount again")
            self._refresh_widget_states()
        elif event.kind == "unparked":
            self._parked = False
            self._park_status_var.set("")
            self._refresh_widget_states()
