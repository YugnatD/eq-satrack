"""Floating finder-camera window -- shows the wide-field finder live
preview + plate-solve / sync controls, reachable from any tab without
leaving the main workflow.

Same show/hide pattern as JogWindow: created once, withdrawn at start,
shown via a button in the Transit tab.  Destroyed only when the app
closes.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from am5.gui.panels import format_exposure_us, show_frame_on_canvas
from am5.gui.theme import PALETTE
from camera.finder import MAX_FINDER_EXPOSURE_US, FinderState
from camera.platesolve import PlateSolver
from camera.worker import CameraEvent, CameraWorker, pgm_to_array


class FinderWindow(tk.Toplevel):
    """Floating live view of the finder camera + plate-solve / sync."""

    def __init__(
        self,
        parent: tk.Misc,
        finder_worker: CameraWorker,
        finder_state: FinderState,
        on_sync,                        # callable(ra_deg, dec_deg) → MountWorker.sync(...)
        fov_deg_var: tk.StringVar | None = None,
    ):
        super().__init__(parent)
        self.title("Finder camera — live view")
        self.configure(background=PALETTE.bg)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self.geometry("520x560")

        self._finder_worker = finder_worker
        self._finder_state = finder_state
        self._on_sync = on_sync
        # SVBony 60mm F/4 + ASI 678MM: FOV ~1.83° × 1.24°, plate scale ~1.72 "/px
        self._fov_deg_var = fov_deg_var or tk.StringVar(value="1.83")
        self._solver = PlateSolver()
        self._latest_frame: np.ndarray | None = None
        self._photo: tk.PhotoImage | None = None
        self._camera_controls: list[str] = []  # control names reported at connection
        self._connected = False

        # -- header status --
        self._status_var = tk.StringVar(value="Finder not connected -- connect in the Connection tab")
        ttk.Label(self, textvariable=self._status_var, foreground=PALETTE.fg_dim).pack(
            anchor="w", padx=10, pady=(8, 0),
        )

        # -- exposure / gain sliders (same log-scale approach as main camera) --
        import math as _math
        self._exp_log = tk.DoubleVar(value=_math.log10(50000))  # 50ms default, ASI 678MM
        self._exp_label = tk.StringVar(value=format_exposure_us(50000))
        self._gain_var = tk.DoubleVar(value=100)
        self._gain_label = tk.StringVar(value="100")

        ctrl_frame = ttk.LabelFrame(self, text="Exposure / gain", padding=6)
        ctrl_frame.pack(fill="x", padx=10, pady=(4, 0))
        exp_row = ttk.Frame(ctrl_frame)
        exp_row.pack(fill="x")
        ttk.Label(exp_row, text="Exp", width=4).pack(side="left")
        self._exp_scale = ttk.Scale(
            exp_row, from_=1.5, to=_math.log10(MAX_FINDER_EXPOSURE_US), variable=self._exp_log, state="disabled",
            command=lambda _v: self._on_slider_change(),
        )
        self._exp_scale.pack(side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Label(exp_row, textvariable=self._exp_label, width=10).pack(side="left")
        gain_row = ttk.Frame(ctrl_frame)
        gain_row.pack(fill="x", pady=(2, 0))
        ttk.Label(gain_row, text="Gain", width=4).pack(side="left")
        self._gain_scale = ttk.Scale(
            gain_row, from_=0, to=570, variable=self._gain_var, state="disabled",
            command=lambda _v: self._on_slider_change(),
        )
        self._gain_scale.pack(side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Label(gain_row, textvariable=self._gain_label, width=6).pack(side="left")

        # -- live preview canvas --
        self._canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=10, pady=(6, 0))

        # -- blob info --
        self._blob_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._blob_var, foreground=PALETTE.accent_ok).pack(
            anchor="w", padx=10,
        )

        # -- plate solve controls --
        solve_frame = ttk.LabelFrame(self, text="Plate solve & sync", padding=8)
        solve_frame.pack(fill="x", padx=10, pady=(6, 8))

        row1 = ttk.Frame(solve_frame)
        row1.pack(anchor="w")
        ttk.Label(row1, text="FOV (deg):").pack(side="left")
        ttk.Entry(row1, textvariable=self._fov_deg_var, width=6).pack(side="left", padx=(4, 12))
        avail = self._solver.available
        state = "normal" if avail else "disabled"
        self._solve_btn = ttk.Button(row1, text="Plate solve", command=self._on_solve, state=state)
        self._solve_btn.pack(side="left")
        self._sync_btn = ttk.Button(row1, text="Sync mount", command=self._on_sync_click, state="disabled")
        self._sync_btn.pack(side="left", padx=(6, 0))

        if not avail:
            ttk.Label(
                solve_frame,
                text="ASTAP not found. Install ASTAP and add it to $PATH to enable plate solving.",
                foreground=PALETTE.accent_warn, wraplength=480, justify="left",
            ).pack(anchor="w", pady=(4, 0))

        self._solve_status_var = tk.StringVar(value="")
        ttk.Label(solve_frame, textvariable=self._solve_status_var, foreground=PALETTE.fg_dim,
                  wraplength=480, justify="left").pack(anchor="w", pady=(4, 0))

        self._solved_ra: float | None = None
        self._solved_dec: float | None = None

    # ------------------------------------------------------------------
    # Public -- called from App._pump_events

    def _on_slider_change(self) -> None:
        self._exp_label.set(format_exposure_us(10 ** self._exp_log.get()))
        self._gain_label.set(str(round(self._gain_var.get())))
        if self._connected:
            self._apply_camera_settings()

    def _apply_camera_settings(self) -> None:
        if not self._connected:
            return
        exp_us = round(10 ** self._exp_log.get())
        gain = round(self._gain_var.get())
        self._finder_worker.set_exposure_us(exp_us)
        self._finder_worker.set_gain(gain)

    def handle_camera_event(self, event: CameraEvent) -> None:
        if event.kind == "connected":
            self._connected = True
            w, h = event.payload["width"], event.payload["height"]
            self._status_var.set(
                f"Finder connected — {w}×{h} "
                f"({'colour' if event.payload['is_color'] else 'mono'})"
            )
            self._exp_scale.configure(state="normal")
            self._gain_scale.configure(state="normal")
            # Push default settings immediately so camera isn't left at power-on defaults
            self._apply_camera_settings()
            # Auto-fill the plate-solve FOV hint from the actually-configured
            # finder optics (ConnectionPanel sets finder_plate_scale_arcsec
            # synchronously before connect, so it's already current here) --
            # still just a starting point the operator can edit, but no
            # longer a static "1.83" disconnected from the real hardware.
            if self._finder_state.finder_plate_scale_arcsec > 0:
                fov_deg = w * self._finder_state.finder_plate_scale_arcsec / 3600.0
                self._fov_deg_var.set(f"{fov_deg:.2f}")
        elif event.kind == "disconnected":
            self._connected = False
            self._status_var.set("Finder not connected -- connect in the Connection tab")
            self._latest_frame = None
            self._exp_scale.configure(state="disabled")
            self._gain_scale.configure(state="disabled")
            # A solved position is only valid for the mount pose it was
            # solved at -- a disconnect means the operator may reconnect
            # later pointed somewhere else entirely, see _invalidate_solve.
            self._invalidate_solve()
        elif event.kind == "preview_frame":
            frame = pgm_to_array(event.payload["pgm"])
            self._latest_frame = frame
            self._finder_state.update_frame(frame)
            if self.winfo_ismapped():
                self._show_preview(frame)

    def handle_mount_event(self, event) -> None:
        """Called from App._pump_events for every MountWorker event (same
        pattern as TransitPanel/JogWindow's own handle_mount_event) --
        invalidates a previously plate-solved sync target the moment the
        mount is explicitly commanded to point somewhere else, so a stale
        solve can't get synced to the wrong place. Only reacts to discrete
        "the mount was told to move" events (a real GOTO, a manual
        jog-to-target, tracking starting), not the continuous "position"
        idle-poll stream, which fires at 2-20Hz regardless of whether the
        mount actually moved and would otherwise disable the sync button
        almost immediately after every solve."""
        if event.kind in ("goto_result", "jog_goto_result", "tracking_started"):
            self._invalidate_solve()

    def _invalidate_solve(self) -> None:
        """Clears a previously plate-solved RA/DEC and disables Sync --
        same rationale as FinderState.reset_blob: a stale detection must
        not be able to keep silently driving a real action (here,
        Mount.sync(), which overwrites the mount's believed position
        WITHOUT moving it -- see am5/mount.py's own docstring) after the
        state that produced it is no longer current."""
        self._solved_ra = None
        self._solved_dec = None
        self._sync_btn.configure(state="disabled")

    # ------------------------------------------------------------------

    def _show_preview(self, frame: np.ndarray) -> None:
        drawn = show_frame_on_canvas(self._canvas, frame)
        if drawn is None:
            return
        self._photo = drawn.photo  # keep a reference -- Tk drops images with none
        scale, xoff, yoff = drawn.scale, drawn.x_offset, drawn.y_offset
        corners = self._finder_state.main_fov_corners_px()
        if corners is not None:
            points = []
            for row, col in corners:
                points.append(int(col * scale) + xoff)
                points.append(int(row * scale) + yoff)
            self._canvas.create_polygon(points, outline="lime", fill="", width=2)
        if self._finder_state.blob_found and self._finder_state.last_blob_row is not None:
            bx = int(self._finder_state.last_blob_col * scale) + xoff
            by = int(self._finder_state.last_blob_row * scale) + yoff
            r = 14
            self._canvas.create_oval(bx - r, by - r, bx + r, by + r, outline="red", width=2)
            self._blob_var.set(
                f"ISS blob: ({self._finder_state.last_blob_col:.0f}, "
                f"{self._finder_state.last_blob_row:.0f}) px"
            )
        else:
            self._blob_var.set("No blob detected")

    def _on_solve(self) -> None:
        if self._latest_frame is None:
            self._solve_status_var.set("No frame yet")
            return
        try:
            fov = float(self._fov_deg_var.get())
        except ValueError:
            self._solve_status_var.set("Invalid FOV value")
            return
        self._solve_btn.configure(state="disabled")
        self._sync_btn.configure(state="disabled")
        self._solve_status_var.set("Solving…")
        frame = self._latest_frame.copy()
        self._solver.solve_async(
            frame, self, self._on_solve_done, fov_deg=fov,
        )

    def _on_solve_done(self, result) -> None:
        self._solve_btn.configure(state="normal" if self._solver.available else "disabled")
        if result.success:
            self._solved_ra = result.ra_deg
            self._solved_dec = result.dec_deg
            self._solve_status_var.set(
                f"✓  RA {result.ra_deg / 15.0:.5f}h  DEC {result.dec_deg:+.4f}°  "
                f"scale {result.pixel_scale_arcsec:.2f}\"/px  rot {result.field_rotation_deg:.1f}°"
            )
            self._sync_btn.configure(state="normal")
        else:
            self._invalidate_solve()
            self._solve_status_var.set(f"✗  {result.message}")

    def _on_sync_click(self) -> None:
        if self._solved_ra is None or self._solved_dec is None:
            return
        self._on_sync(self._solved_ra, self._solved_dec)
        self._solve_status_var.set(
            f"Sync sent — RA {self._solved_ra / 15.0:.5f}h  DEC {self._solved_dec:+.4f}°"
        )
        self._sync_btn.configure(state="disabled")
