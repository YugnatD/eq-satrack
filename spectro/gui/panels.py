"""Frontend for the Star Analyser spectroscopy app.

TargetPanel is wired to a real star catalog (spectro/catalog.py, SIMBAD via
astroquery) -- target search and standard-star candidates are real lookups,
not demo data. AcquisitionPanel/FlatsPanel generate synthetic-but-realistic
frames (no real camera connected yet -- see Connection tab, still a pure
mock) but ReductionPanel and SpectrumPanel run the REAL reduction pipeline
(spectro/reduction.py: stacking, dark/flat calibration, line detection,
dispersion fitting, instrument-response flux calibration) on those frames,
not a canned demo -- the same code would run unchanged on real captures.
Reuses am5.gui.theme (PALETTE / style_axes) rather than the mount-specific
panels, so the two apps look like one product without sharing any
tracking-specific code.
"""

from __future__ import annotations

import csv
import math
import queue
import threading
import tkinter as tk
from datetime import datetime, timezone
from tkinter import filedialog, ttk

import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from am5.angles import angular_separation_deg
from am5.gui.theme import PALETTE, style_axes
from am5.gui.worker import MountWorker, WorkerEvent
from spectro.avspec_export import AvspecExportError, write_avspec_fits
from spectro.catalog import (
    REFERENCE_LINES,
    Star,
    StarNotFound,
    altitude_track,
    estimate_teff_k,
    find_standard_candidates,
    is_standard_candidate,
    model_spectrum,
    resolve_target,
)
from spectro.pickles import FetchError, fetch_reference_spectrum
from spectro.reduction import (
    ASSUMED_WL_MAX,
    ASSUMED_WL_MIN,
    PROFILE_HALF_HEIGHT,
    TRAIL_END_PX,
    TRAIL_ROW,
    TRAIL_START_PX,
    ReductionError,
    apply_response,
    calibrate_science,
    compute_response,
    detect_line_pixels,
    extract_profile,
    fit_dispersion,
    px_for_wavelength,
    resolution_at,
    snr_gain,
    stack_frames,
    wavelength_for_px,
)

def format_exposure_us(microseconds: float) -> str:
    if microseconds >= 1_000_000:
        return f"{microseconds / 1_000_000:.2f} s"
    if microseconds >= 1000:
        return f"{microseconds / 1000:.2f} ms"
    return f"{microseconds:.0f} us"


# -- demo data ---------------------------------------------------------------

# Spectral lines a Star Analyser (R~100-500) can plausibly resolve -- used
# to make the mockup plots/labels look like real output, not placeholders.
DEMO_LINES = [
    ("Ca II K", 3934), ("Ca II H", 3968), ("Hδ", 4102), ("Hγ", 4340),
    ("Hβ", 4861), ("Na D", 5893), ("Hα", 6563),
]

# Geometry is owned by spectro/reduction.py now (shared with the real
# reduction pipeline, which needs the exact same trail/pixel mapping this
# mock paints its frames with) -- local aliases so the rest of this file
# doesn't need a mass rename.
_TRAIL_ROW = TRAIL_ROW
_TRAIL_START_PX = TRAIL_START_PX
_TRAIL_END_PX = TRAIL_END_PX
_PROFILE_HALF_HEIGHT = PROFILE_HALF_HEIGHT
_ASSUMED_WL_MIN, _ASSUMED_WL_MAX = ASSUMED_WL_MIN, ASSUMED_WL_MAX
_assumed_px_for_wavelength = px_for_wavelength
_wavelength_for_px = wavelength_for_px
_extract_profile = extract_profile

# Where the star is assumed to sit once properly framed (roi_offset=0,0 in
# _synthetic_trail_image), and how close is "close enough" -- shared by
# AcquisitionPanel's centering check AND _draw_center_guide's on-image
# target marker, so the guide always matches what actually counts as
# centered rather than two numbers drifting apart.
_ROI_TARGET_X, _ROI_TARGET_Y = 55, _TRAIL_ROW
_ROI_TOLERANCE_X, _ROI_TOLERANCE_Y = 8, 4

# How often the live preview redraws itself with a fresh noise/jitter
# frame -- ~7fps, plausible for a live view at these exposures and fast
# enough to actually read as "live" rather than a static plot, without
# redrawing matplotlib canvases so often it competes with slider drags
# for CPU. Same self-paced self.after() idiom as am5/gui/panels.py's SER
# player loop (_play_tick), since this mockup has no real camera worker
# thread yet to push frames on its own schedule.
_LIVE_INTERVAL_MS = 140


# Fixed (not re-randomized on every redraw) relative line depths/widths for
# the mock preview's absorption dips -- loosely realistic for an A-type
# star (Balmer series deepens toward Hα, Ca II shallower) rather than
# scientifically derived, but consistent every time so the operator learns
# to recognize one real-looking pattern instead of a different fake one on
# every slider tick. Only used for the LIVE PREVIEW mock (this tab's whole
# point is "does this look like it's working") -- never for the actually-
# fetched Pickles/model data shown in the Target & standard tab, which is
# real (see spectro/pickles.py).
_LINE_DEPTH = {"Ca II K": 0.22, "Ca II H": 0.20, "Hδ": 0.32, "Hγ": 0.38, "Hβ": 0.45, "Na D": 0.15, "Hα": 0.50}
_LINE_WIDTH_A = {"Ca II K": 7.0, "Ca II H": 7.0, "Hδ": 9.0, "Hγ": 10.0, "Hβ": 12.0, "Na D": 6.0, "Hα": 13.0}


def _apply_reference_dips(wl: np.ndarray, flux: np.ndarray) -> np.ndarray:
    out = flux.copy()
    for label, line_wl in REFERENCE_LINES:
        depth = _LINE_DEPTH.get(label, 0.2)
        width = _LINE_WIDTH_A.get(label, 8.0)
        out = out * (1.0 - depth * np.exp(-((wl - line_wl) ** 2) / (2 * width**2)))
    return out


def _synthetic_spectrum(seed: int, teff: float = 7200.0) -> tuple[np.ndarray, np.ndarray]:
    """Fallback continuum+lines for when no real/estimated star is picked
    yet -- real wavelength-domain Planck law (catalog.model_spectrum, the
    same physics used everywhere else in this app) plus the same fixed
    line-dip profile applied to every other live-preview mock, so this
    fallback looks like a plausible member of the same family of spectra
    rather than a visibly different placeholder shape."""
    rng = np.random.default_rng(seed)
    wl, flux = model_spectrum(teff, wl_min=_ASSUMED_WL_MIN, wl_max=_ASSUMED_WL_MAX, n=900)
    flux = _apply_reference_dips(wl, flux)
    flux = flux + rng.normal(0.0, 0.012, size=wl.shape)
    return wl, flux


def _synthetic_trail_image(
    seed: int, brightness_scale: float = 1.0, spectrum: tuple[np.ndarray, np.ndarray] | None = None,
    frame_seed: int | None = None, roi_offset_x: float = 0.0, roi_offset_y: float = 0.0,
    include_signal: bool = True,
) -> np.ndarray:
    """Order-0 star + order-1 spectral trail, shaped like a real CMOS
    frame rather than flat noise everywhere: a read-noise floor, shot
    noise that scales with local signal (brighter pixels are grainier,
    same as a real sensor), and the occasional hot pixel. `spectrum`, if
    given, is the actual (wavelength, normalized flux) curve to paint
    onto the trail -- e.g. the real Pickles data already fetched for the
    selected reference star (see TargetPanel.get_reference_spectrum) --
    instead of the generic fallback shape, so a reference star's preview
    mock actually resembles that star's real spectrum rather than an
    unrelated placeholder curve.

    `frame_seed`, if given, drives ONLY the noise/jitter draw (not which
    star/spectrum is shown, which stays keyed on `seed`) -- passing a
    fresh frame_seed every ~100ms is what makes a continuously-called
    live preview loop actually look alive (new grain, a slightly
    trembling star position, like real atmospheric seeing) instead of a
    perfectly frozen image between slider moves.

    `roi_offset_x/y` shift BOTH the star blob and the trail together, as
    real ROI panning would -- (0, 0) reproduces the "properly framed"
    layout every other function here assumes (_TRAIL_START_PX etc.), a
    nonzero offset simulates imperfect GOTO pointing: the star (and its
    trail) sit off where they're assumed to be until the operator pans
    the ROI to bring them back to (0, 0), see AcquisitionPanel's ROI
    controls.

    `include_signal=False` skips the star blob AND the trail entirely --
    what a real DARK or OFFSET/BIAS frame looks like (cap on, no optical
    signal at all, only the sensor's own bias/read/shot noise) -- used by
    AcquisitionPanel's dark/offset capture instead of reusing a science
    frame with the star just left in."""
    noise_rng = np.random.default_rng(frame_seed if frame_seed is not None else seed)
    h, w = 90, 420
    bias, read_noise = 8.0, 3.0
    img = noise_rng.normal(bias, read_noise, size=(h, w))

    if include_signal:
        order0_x = 55 + roi_offset_x + noise_rng.normal(0.0, 0.6)  # seeing/tracking jitter, a pixel or so
        order0_y = h / 2.0 + roi_offset_y + noise_rng.normal(0.0, 0.5)
        yy, xx = np.mgrid[0:h, 0:w]
        img += 220.0 * brightness_scale * np.exp(-(((xx - order0_x) ** 2 + (yy - order0_y) ** 2) / (2 * 3.5**2)))

        wl, flux = spectrum if spectrum is not None else _synthetic_spectrum(seed)
        trail_wl = _wavelength_for_px(xx - roi_offset_x)
        trail_flux = np.interp(trail_wl.ravel(), wl, flux, left=0.0, right=0.0).reshape(xx.shape)
        img += np.where(
            xx >= _TRAIL_START_PX + roi_offset_x,
            trail_flux * 150.0 * brightness_scale * np.exp(-((yy - order0_y) ** 2) / (2 * 4.0**2)),
            0.0,
        )

    # Shot noise: real sensors get grainier where the signal is brighter
    # (Poisson statistics), not a uniform noise floor everywhere.
    signal_above_bias = np.clip(img - bias, 0.0, None)
    img = img + noise_rng.normal(0.0, np.sqrt(signal_above_bias + 1.0)) * 0.5

    for _ in range(noise_rng.integers(0, 3)):
        hx, hy = int(noise_rng.integers(0, w)), int(noise_rng.integers(0, h))
        img[hy, hx] = 255.0

    return np.clip(img, 0, 255)


def _synthetic_flat_image(seed: int, brightness_pct: float, frame_seed: int | None = None) -> np.ndarray:
    """Evenly-illuminated field (twilight sky / flat panel) with mild
    vignetting and shot noise that scales with brightness (same idea as
    _synthetic_trail_image) -- brightness_pct (0-100) scales the mean
    level, so tuning the exposure slider visibly moves the histogram, the
    same way tuning real exposure against a real flat panel does.
    `frame_seed`, if given, drives the noise draw for a live-look preview
    (see _synthetic_trail_image's own docstring)."""
    rng = np.random.default_rng(frame_seed if frame_seed is not None else seed)
    h, w = 90, 420
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    r = np.hypot((xx - cx) / cx, (yy - cy) / cy)
    vignette = 1.0 - 0.18 * r**2
    signal = 255.0 * (brightness_pct / 100.0) * vignette
    shot = rng.normal(0.0, np.sqrt(np.clip(signal, 0.0, None) + 1.0)) * 0.3
    img = signal + shot + rng.normal(0.0, 2.0, size=(h, w))
    return np.clip(img, 0, 255)


def _draw_profile(ax, figure, image: np.ndarray) -> None:
    """Uncalibrated profile (raw pixel counts vs. pixel position) with the
    real reference lines marked at their ASSUMED pixel position (see
    _assumed_px_for_wavelength) -- NOT a calibrated spectrum, just a live
    sanity check that the trail actually looks like a stellar spectrum
    (dips roughly where the Balmer series etc. should be) while framing/
    focusing, well before the real wavelength calibration step in the
    Spectrum tab."""
    ax.clear()
    profile = _extract_profile(image)
    pixels = np.arange(len(profile))
    ax.plot(pixels[_TRAIL_START_PX:], profile[_TRAIL_START_PX:], color=PALETTE.accent, linewidth=1)
    for label, wl in REFERENCE_LINES:
        px = _assumed_px_for_wavelength(wl)
        if _TRAIL_START_PX <= px <= _TRAIL_END_PX:
            ax.axvline(px, color=PALETTE.border, linewidth=0.8)
            ax.text(
                px, 1.02, label, color=PALETTE.fg_dim, fontsize=6, ha="center", va="bottom",
                transform=ax.get_xaxis_transform(),
            )
    ax.set_xlabel("pixel (assumed dispersion, not yet calibrated)", fontsize=8)
    ax.set_ylabel("counts", fontsize=8)
    style_axes(figure, ax)


def _draw_center_guide(ax, centered: bool) -> None:
    """Overlay on the live preview showing WHERE to put the star -- a
    crosshair + tolerance box at the assumed-framing position
    (_ROI_TARGET_X/Y) -- so "pan the ROI until it's centered" has an
    actual visible target instead of the operator having to guess from
    the (unreadable, once off) profile plot alone. Color follows the same
    centered/not-centered state as the status label above it."""
    color = PALETTE.accent_ok if centered else PALETTE.accent_warn
    ax.plot(_ROI_TARGET_X, _ROI_TARGET_Y, marker="+", color=color, markersize=12, markeredgewidth=1.5)
    ax.add_patch(
        Rectangle(
            (_ROI_TARGET_X - _ROI_TOLERANCE_X, _ROI_TARGET_Y - _ROI_TOLERANCE_Y),
            2 * _ROI_TOLERANCE_X, 2 * _ROI_TOLERANCE_Y,
            fill=False, edgecolor=color, linewidth=1, linestyle="--",
        ),
    )


def _draw_extraction_band(ax, w: int) -> None:
    """Outlines exactly which rows get summed into the extracted profile
    (see _extract_profile) -- without this, it's not obvious the profile
    plot only reads a thin band around the trail rather than the whole
    frame, which matters once the trail isn't sitting where it's assumed
    to (see _draw_center_guide, same underlying problem: nothing on the
    image itself showed where the "correct" position actually was)."""
    y0, y1 = _TRAIL_ROW - _PROFILE_HALF_HEIGHT, _TRAIL_ROW + _PROFILE_HALF_HEIGHT
    ax.add_patch(Rectangle((0, y0), w, y1 - y0, fill=False, edgecolor=PALETTE.accent, linewidth=0.8, linestyle=":"))


def _draw_histogram(ax, figure, image: np.ndarray, compact: bool = False) -> None:
    """Shared live-histogram renderer -- AcquisitionPanel (a small,
    discreet version -- compact=True -- for a quick over/under-exposure
    glance; the uncalibrated profile is that tab's primary readout, see
    _draw_profile) and FlatsPanel (full-size, where hitting the ~2/3 full
    well line via this readout is the whole point) both use this."""
    ax.clear()
    counts, edges = np.histogram(image.flatten(), bins=32, range=(0, 255))
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=PALETTE.accent)
    two_thirds = 255 * 2.0 / 3.0
    ax.axvline(two_thirds, color=PALETTE.accent_warn, linestyle="--", linewidth=1)
    ax.set_xlim(0, 255)
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.text(two_thirds, ax.get_ylim()[1] if ax.get_ylim()[1] else 1, "2/3", color=PALETTE.accent_warn, fontsize=7, ha="center", va="bottom")
        ax.set_xlabel("pixel value", fontsize=8)
        ax.set_ylabel("count", fontsize=8)
    style_axes(figure, ax)


# -- shared bits ---------------------------------------------------------------


class MockDeviceRow(ttk.Frame):
    """One "Connect" row (mount or camera) -- pure UI state, no worker."""

    def __init__(self, parent: tk.Misc, title: str, detail_lines: list[str]):
        super().__init__(parent)
        self._connected = False
        frame = ttk.LabelFrame(self, text=title, padding=8)
        frame.pack(fill="x")

        self._kind_var = tk.StringVar(value="mock")
        ttk.Radiobutton(frame, text="Mock", variable=self._kind_var, value="mock").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Real", variable=self._kind_var, value="real").grid(row=0, column=1, sticky="w")

        self._connect_button = ttk.Button(frame, text="Connect", command=self._on_connect)
        self._connect_button.grid(row=1, column=0, pady=(6, 0))
        self._disconnect_button = ttk.Button(frame, text="Disconnect", command=self._on_disconnect, state="disabled")
        self._disconnect_button.grid(row=1, column=1, pady=(6, 0))

        self._status_var = tk.StringVar(value="Not connected")
        ttk.Label(frame, textvariable=self._status_var, foreground=PALETTE.fg_dim).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0),
        )
        for line in detail_lines:
            ttk.Label(frame, text=line, foreground=PALETTE.fg_dim).grid(
                row=3 + detail_lines.index(line), column=0, columnspan=2, sticky="w",
            )

    def get_kind(self) -> str:
        return self._kind_var.get()

    def _on_connect(self) -> None:
        self._connected = True
        self._status_var.set(f"Connected ({self._kind_var.get()})")
        self._status_var_color("ok")
        self._connect_button.configure(state="disabled")
        self._disconnect_button.configure(state="normal")

    def _on_disconnect(self) -> None:
        self._connected = False
        self._status_var.set("Not connected")
        self._connect_button.configure(state="normal")
        self._disconnect_button.configure(state="disabled")

    def _status_var_color(self, _kind: str) -> None:
        pass  # placeholder hook -- real version would recolor the status label


# -- Connection tab ---------------------------------------------------------------


class ConnectionPanel(ttk.Frame):
    """Mount connection is REAL -- this panel owns a real
    am5.gui.worker.MountWorker (unchanged from the ISS tracker), talking
    to either a MockMount or actual serial hardware exactly the way that
    project's own ConnectionPanel does. Manual jog itself lives in a
    separate floating window (spectro/gui/jog_window.py, owned by App,
    same shown-not-destroyed pattern as the ISS tracker's own JogWindow)
    rather than embedded here, so it's reachable from any tab -- see
    on_connection_change, which App wires to that window's
    set_connected(). Camera is still a pure visual mock (see
    MockDeviceRow) -- real camera wiring hasn't been requested for this
    app yet, only telescope movement."""

    def __init__(self, parent: tk.Misc, mount_worker: MountWorker, on_connection_change=None):
        super().__init__(parent, padding=10)
        self._mount_worker = mount_worker
        self._on_connection_change = on_connection_change
        columns = ttk.Frame(self)
        columns.pack(fill="both", expand=True)
        left = ttk.Frame(columns)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(columns)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self._build_mount_control(left)
        self._camera_row = MockDeviceRow(left, "Camera (ASI290MC + Star Analyser)", ["Reuses camera/ -- unchanged"])
        self._camera_row.pack(fill="x", pady=(10, 0))

        site_frame = ttk.LabelFrame(right, text="Observation site", padding=8)
        site_frame.pack(fill="x")
        self._site_lat_var = tk.StringVar(value="46.18")
        self._site_lon_var = tk.StringVar(value="6.14")
        self._site_elevation_var = tk.StringVar(value="400")
        for i, (label, var) in enumerate((
            ("lat", self._site_lat_var), ("lon", self._site_lon_var), ("elevation (m)", self._site_elevation_var),
        )):
            ttk.Label(site_frame, text=label).grid(row=i, column=0, sticky="w")
            ttk.Entry(site_frame, textvariable=var, width=10).grid(row=i, column=1, sticky="w")

        grating_frame = ttk.LabelFrame(right, text="Instrument", padding=8)
        grating_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(grating_frame, text="Grating").grid(row=0, column=0, sticky="w")
        grating_var = tk.StringVar(value="Star Analyser SA-200 (200 l/mm)")
        ttk.Combobox(
            grating_frame, textvariable=grating_var, state="readonly", width=28,
            values=["Star Analyser SA-100 (100 l/mm)", "Star Analyser SA-200 (200 l/mm)"],
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(grating_frame, text="Focal length (mm)").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(grating_frame, width=10).grid(row=1, column=1, sticky="w", pady=(4, 0))

    def _build_mount_control(self, parent: tk.Misc) -> None:
        frame = ttk.LabelFrame(parent, text="Mount (AM3/AM5)", padding=8)
        frame.pack(fill="x")

        self._mount_kind_var = tk.StringVar(value="mock")
        self._mount_address_var = tk.StringVar(value="/dev/ttyACM0")
        self._mount_seed_var = tk.StringVar(value="")
        for i, (label, value) in enumerate((("Mock", "mock"), ("Serial", "serial"), ("TCP", "tcp"))):
            ttk.Radiobutton(
                frame, text=label, variable=self._mount_kind_var, value=value,
                command=self._update_mount_address_state,
            ).grid(row=0, column=i, sticky="w")
        self._mount_address_entry = ttk.Entry(frame, textvariable=self._mount_address_var, width=18)
        self._mount_address_entry.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(frame, text="port / host:port", foreground=PALETTE.fg_dim).grid(
            row=1, column=2, sticky="w", pady=(4, 0),
        )
        ttk.Label(frame, text="mock seed (optional)", foreground=PALETTE.fg_dim).grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self._mount_seed_var, width=8).grid(row=2, column=1, sticky="w")
        self._update_mount_address_state()

        self._mount_connect_button = ttk.Button(frame, text="Connect", command=self._on_mount_connect)
        self._mount_connect_button.grid(row=3, column=0, pady=(8, 0))
        self._mount_disconnect_button = ttk.Button(
            frame, text="Disconnect", command=self._mount_worker.disconnect, state="disabled",
        )
        self._mount_disconnect_button.grid(row=3, column=1, pady=(8, 0))
        self._mount_status_var = tk.StringVar(value="Not connected")
        ttk.Label(frame, textvariable=self._mount_status_var, foreground=PALETTE.fg_dim).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(6, 0),
        )
        ttk.Label(
            frame, text="Manual jog control is in its own window -- see the\n\"Jog control...\" button at the bottom of the app.",
            foreground=PALETTE.fg_dim, justify="left",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _update_mount_address_state(self) -> None:
        self._mount_address_entry.configure(state="disabled" if self._mount_kind_var.get() == "mock" else "normal")

    def _on_mount_connect(self) -> None:
        seed_text = self._mount_seed_var.get().strip()
        try:
            latitude_deg, longitude_deg = self.get_site_lat_deg(), self.get_site_lon_deg()
        except ValueError:
            self._mount_status_var.set("Invalid site lat/lon")
            return
        self._mount_connect_button.configure(state="disabled")
        self._mount_status_var.set("Connecting...")
        self._mount_worker.connect(
            self._mount_kind_var.get(), address=self._mount_address_var.get(),
            mock_seed=int(seed_text) if seed_text else None,
            latitude_deg=latitude_deg, longitude_deg=longitude_deg,
        )

    def get_site_lat_deg(self) -> float:
        return float(self._site_lat_var.get())

    def get_site_lon_deg(self) -> float:
        return float(self._site_lon_var.get())

    def get_site_elevation_m(self) -> float:
        return float(self._site_elevation_var.get())

    def is_mock(self) -> bool:
        """True if EITHER device is set to Mock -- used to refuse exports
        meant for real submission (see SpectrumPanel's AVSpec export),
        so a mock run can't produce a file that looks like a real
        observation. Camera is always effectively mock right now (no real
        CameraWorker exists yet, see this module's docstring), but the
        radio button still reflects operator intent and is checked here
        for when that changes."""
        return self._mount_kind_var.get() == "mock" or self._camera_row.get_kind() == "mock"

    def handle_mount_event(self, event: WorkerEvent) -> None:
        if event.kind == "connected":
            self._mount_status_var.set(f"Connected -- firmware {event.payload['firmware']}")
            self._mount_disconnect_button.configure(state="normal")
            if self._on_connection_change is not None:
                self._on_connection_change(True)
        elif event.kind == "connect_error":
            self._mount_status_var.set(f"Connection failed: {event.payload['message']}")
            self._mount_connect_button.configure(state="normal")
        elif event.kind == "disconnected":
            self._mount_status_var.set("Not connected")
            self._mount_connect_button.configure(state="normal")
            self._mount_disconnect_button.configure(state="disabled")
            if self._on_connection_change is not None:
                self._on_connection_change(False)


# -- Target / standard star tab ---------------------------------------------------------------


class TargetPanel(ttk.Frame):
    """Real SIMBAD lookups (spectro/catalog.py) -- search runs on a
    background thread (network I/O, can take seconds or fail) and posts
    results back through self._results, polled by self.after(), the same
    pattern am5/gui/panels.py's PassesPanel uses for its own background
    TLE/trajectory work. Never touch Tk widgets from the worker thread."""

    # TODO: pull from the Connection tab's site fields once that's wired
    # to a shared SiteVars-like object (see am5/gui/panels.py's own
    # SiteVars docstring for why a single shared source of truth matters
    # -- two independent copies drifting apart bit this project once
    # already). Hardcoded to ConnectionPanel's own displayed defaults for
    # now so at least the two don't visibly disagree.
    _SITE_LAT_DEG = 46.18
    _SITE_LON_DEG = 6.14
    _SITE_ELEVATION_M = 400.0

    def __init__(self, parent: tk.Misc):
        super().__init__(parent, padding=10)
        self._target: Star | None = None
        self._results: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._spectrum_request_id = 0
        self._reference_spectrum: tuple[np.ndarray, np.ndarray] | None = None

        search_row = ttk.Frame(self)
        search_row.pack(fill="x")
        ttk.Label(search_row, text="Target star").pack(side="left")
        self._search_var = tk.StringVar(value="Regulus")
        entry = ttk.Entry(search_row, textvariable=self._search_var, width=24)
        entry.pack(side="left", padx=(8, 4))
        entry.bind("<Return>", lambda _e: self._on_search())
        self._search_button = ttk.Button(search_row, text="Search", command=self._on_search)
        self._search_button.pack(side="left")

        self._target_info_var = tk.StringVar(value="Search a target (resolved via SIMBAD) to see its details here.")
        ttk.Label(self, textvariable=self._target_info_var, justify="left", foreground=PALETTE.fg_dim).pack(
            anchor="w", pady=(8, 0),
        )

        columns = ttk.Frame(self)
        columns.pack(fill="both", expand=True, pady=(12, 0))
        left = ttk.Frame(columns)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(columns)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        ttk.Label(
            left, foreground=PALETTE.fg_dim, wraplength=440, justify="left",
            text=(
                "A spectrophotometric standard star -- NOT a second target -- with a well-known "
                "reference spectrum, used to correct your instrument's own response (sensor + optics + "
                "atmosphere). Pick one close in the sky and at a similar altitude to your target, so "
                "the correction stays valid for both."
            ),
        ).pack(anchor="w", pady=(0, 8))

        self._cand_frame = ttk.LabelFrame(left, text="Standard star candidates (angular separation + airmass match, now)", padding=8)
        cand_frame = self._cand_frame
        cand_frame.pack(fill="both", expand=True)
        columns_spec = ("name", "type", "vmag", "sep", "airmass")
        self._tree = ttk.Treeview(cand_frame, columns=columns_spec, show="headings", height=6, selectmode="browse")
        for col, label, width in (
            ("name", "Name", 90), ("type", "Sp. type", 70), ("vmag", "V mag", 55),
            ("sep", "Sep (deg)", 75), ("airmass", "Δairmass", 80),
        ):
            self._tree.heading(col, text=label)
            self._tree.column(col, width=width, anchor="center")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_candidate_selected)
        self._candidates: list = []

        self._recommend_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self._recommend_var, foreground=PALETTE.accent_ok, wraplength=440, justify="left").pack(
            anchor="w", pady=(6, 0),
        )

        chart_frame = ttk.LabelFrame(right, text="Altitude, next few hours -- target vs. chosen standard", padding=8)
        chart_frame.pack(fill="both", expand=True)
        self._chart_figure = Figure(figsize=(4, 2.6), dpi=100)
        self._chart_ax = self._chart_figure.add_subplot(111, projection="polar")
        self._chart_canvas = FigureCanvasTkAgg(self._chart_figure, master=chart_frame)
        self._chart_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._reset_chart()

        self._spectrum_frame = ttk.LabelFrame(right, text="Reference spectrum (model)", padding=8)
        self._spectrum_frame.pack(fill="both", expand=True, pady=(10, 0))
        self._spectrum_figure = Figure(figsize=(4, 1.9), dpi=100)
        self._spectrum_ax = self._spectrum_figure.add_subplot(111)
        self._spectrum_canvas = FigureCanvasTkAgg(self._spectrum_figure, master=self._spectrum_frame)
        self._spectrum_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.after(100, self._poll_results)

    def _reset_chart(self) -> None:
        self._chart_ax.clear()
        self._chart_ax.set_theta_zero_location("N")
        self._chart_ax.set_theta_direction(-1)
        self._chart_ax.set_ylim(0, 90)
        self._chart_ax.set_yticks([0, 30, 60, 90])
        self._chart_ax.set_yticklabels(["90°", "60°", "30°", "0°"], fontsize=7)
        style_axes(self._chart_figure, self._chart_ax)
        self._chart_canvas.draw()

    def _on_search(self) -> None:
        name = self._search_var.get().strip()
        if not name:
            return
        self._search_button.configure(state="disabled")
        self._target_info_var.set(f"Searching SIMBAD for {name!r}...")
        self._tree.delete(*self._tree.get_children())
        self._recommend_var.set("")
        threading.Thread(target=self._search_thread, args=(name,), daemon=True).start()

    def _search_thread(self, name: str) -> None:
        try:
            target = resolve_target(name)
            candidates = find_standard_candidates(target, self._SITE_LAT_DEG, self._SITE_LON_DEG, self._SITE_ELEVATION_M)
            self._results.put(("ok", (target, candidates)))
        except StarNotFound as exc:
            self._results.put(("error", str(exc)))
        except Exception as exc:  # noqa: BLE001 - network/SIMBAD hiccups shouldn't crash the app
            self._results.put(("error", f"Lookup failed: {exc}"))

    def _poll_results(self) -> None:
        try:
            kind, payload = self._results.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_results)
            return
        if kind in ("ok", "error"):
            self._search_button.configure(state="normal")
            if kind == "error":
                self._target_info_var.set(f"Error: {payload}")
            else:
                target, candidates = payload
                self._show_target(target, candidates)
        else:
            self._handle_spectrum_result(kind, payload)
        self.after(100, self._poll_results)

    def _show_target(self, target: Star, candidates: list) -> None:
        self._target = target
        coord_str = f"RA {target.ra_deg / 15.0:.4f}h  DEC {target.dec_deg:+.4f} deg"
        vmag_str = f"V={target.vmag:.2f}" if target.vmag is not None else "V=?"
        self._target_info_var.set(f"{target.name}  --  {coord_str}  --  {vmag_str}  --  {target.spectral_type or '?'}")

        self._candidates = candidates
        self._tree.delete(*self._tree.get_children())
        for i, cand in enumerate(candidates):
            airmass_str = f"{cand.airmass_delta:.2f}" if cand.airmass_delta is not None else "below horizon"
            self._tree.insert(
                "", "end", iid=str(i),
                values=(cand.star.name, cand.star.spectral_type, f"{cand.star.vmag:.2f}", f"{cand.separation_deg:.1f}", airmass_str),
            )

        if is_standard_candidate(target):
            # The target already IS a flux standard (e.g. Vega) -- it
            # fills the "known reference" role for itself, no companion
            # star needed. Candidates (if any) are shown as optional
            # alternatives, not auto-selected -- see the conversation this
            # distinction came from: searching Vega/Deneb previously
            # recommended a companion even though neither needs one.
            self._cand_frame.configure(text="Optional alternatives (not needed -- see below)")
            self._recommend_var.set(
                f"{target.name} is itself a spectrophotometric standard ({target.spectral_type or '?'}, {vmag_str}) "
                f"-- calibrate directly from its own reference spectrum, no companion star required.",
            )
            self._draw_chart(target, None)
            self._request_spectrum(target)
            return

        self._cand_frame.configure(text="Standard star candidates (angular separation + airmass match, now)")
        if not candidates:
            self._recommend_var.set(
                "No A0-A3 standard brighter than V=6.5 found within 20 deg -- widen the search or pick one by hand.",
            )
            self._draw_chart(target, None)
            self._clear_spectrum()
            return
        self._tree.selection_set("0")
        self._show_recommendation(candidates[0])
        self._draw_chart(target, candidates[0].star)
        self._request_spectrum(candidates[0].star)

    def _on_candidate_selected(self, _event: object) -> None:
        selection = self._tree.selection()
        if not selection or self._target is None:
            return
        cand = self._candidates[int(selection[0])]
        if not is_standard_candidate(self._target):
            self._show_recommendation(cand)
            self._draw_chart(self._target, cand.star)
        self._request_spectrum(cand.star)

    def _show_recommendation(self, cand) -> None:
        airmass_note = f"airmass matches within {cand.airmass_delta:.2f}" if cand.airmass_delta is not None else "below the horizon right now"
        self._recommend_var.set(
            f"{cand.star.name} ({cand.star.spectral_type}) -- {cand.separation_deg:.1f}° away, {airmass_note}.",
        )

    def _clear_spectrum(self) -> None:
        self._spectrum_ax.clear()
        style_axes(self._spectrum_figure, self._spectrum_ax)
        self._spectrum_frame.configure(text="Reference spectrum")
        self._spectrum_canvas.draw()
        self._reference_spectrum = None

    def _request_spectrum(self, star: Star) -> None:
        """Fetches the REAL Pickles (1998) template spectrum for `star`
        on a background thread (network I/O -- a few hundred ms to a few
        seconds) -- see spectro/pickles.py's module docstring for why
        this is real archival data, not a synthetic approximation. Falls
        back to the blackbody model (spectro/catalog.py's model_spectrum)
        only if no template matches the spectral type, or the fetch
        itself fails (network down, CDS unreachable) -- never silently;
        the plot title always says which one is actually shown.

        self._spectrum_request_id guards against an out-of-order reply:
        clicking through several candidates quickly fires several fetches
        that can complete in any order -- only the reply matching the
        MOST RECENT request is ever applied to the plot."""
        self._spectrum_request_id += 1
        request_id = self._spectrum_request_id
        self._spectrum_frame.configure(text=f"Reference spectrum -- fetching real data for {star.name}...")
        threading.Thread(target=self._spectrum_thread, args=(star, request_id), daemon=True).start()

    def _spectrum_thread(self, star: Star, request_id: int) -> None:
        try:
            result = fetch_reference_spectrum(star.spectral_type)
        except FetchError as exc:
            self._results.put(("spectrum_fetch_error", (star, request_id, str(exc))))
            return
        if result is None:
            self._results.put(("spectrum_no_template", (star, request_id)))
            return
        template_name, wl, flux = result
        self._results.put(("spectrum_ok", (star, request_id, template_name, wl, flux)))

    def _handle_spectrum_result(self, kind: str, payload: tuple) -> None:
        star, request_id = payload[0], payload[1]
        if request_id != self._spectrum_request_id:
            return  # superseded by a newer selection -- discard
        if kind == "spectrum_ok":
            _, _, template_name, wl, flux = payload
            self._render_spectrum(star, wl, flux, f"Pickles (1998) template {template_name!r} -- real archival spectrum")
        else:
            # Real data unavailable (no matching template, or the fetch
            # itself failed) -- fall back to the labeled blackbody model
            # rather than leaving the panel blank.
            teff = estimate_teff_k(star.spectral_type)
            if teff is None:
                self._clear_spectrum()
                self._spectrum_frame.configure(text=f"Reference spectrum -- no data or model for {star.name} ({star.spectral_type or '?'})")
                return
            wl, flux = model_spectrum(teff)
            reason = "no matching Pickles template" if kind == "spectrum_no_template" else f"fetch failed: {payload[2]}"
            self._render_spectrum(star, wl, flux, f"blackbody MODEL (Teff≈{teff:.0f} K, {reason}) -- NOT a measured spectrum")

    def _render_spectrum(self, star: Star, wl: np.ndarray, flux: np.ndarray, source_label: str) -> None:
        self._reference_spectrum = (wl, flux)
        self._spectrum_ax.clear()
        if len(wl) > 0:
            self._spectrum_ax.plot(wl, flux, color=PALETTE.accent, linewidth=1)
            for label, line_wl in REFERENCE_LINES:
                if wl.min() < line_wl < wl.max():
                    self._spectrum_ax.axvline(line_wl, color=PALETTE.border, linewidth=0.8)
                    self._spectrum_ax.text(line_wl, 1.03, label, color=PALETTE.fg_dim, fontsize=6, ha="center")
            self._spectrum_ax.set_xlabel("wavelength (Å)", fontsize=8)
            self._spectrum_ax.set_ylabel("flux (norm.)", fontsize=8)
        self._spectrum_frame.configure(text=f"{star.name} -- {source_label}")
        style_axes(self._spectrum_figure, self._spectrum_ax)
        self._spectrum_canvas.draw()

    def _draw_chart(self, target: Star, standard: Star | None) -> None:
        self._reset_chart()
        when = datetime.now(timezone.utc)
        target_az, target_alt = altitude_track(target, self._SITE_LAT_DEG, self._SITE_LON_DEG, self._SITE_ELEVATION_M, when)
        self._chart_ax.plot(
            [math.radians(a) for a in target_az], [90 - alt for alt in target_alt],
            color=PALETTE.accent, label="Target",
        )
        if standard is not None:
            std_az, std_alt = altitude_track(standard, self._SITE_LAT_DEG, self._SITE_LON_DEG, self._SITE_ELEVATION_M, when)
            self._chart_ax.plot(
                [math.radians(a) for a in std_az], [90 - alt for alt in std_alt],
                color=PALETTE.accent_ok, label="Standard", linestyle="--",
            )
        self._chart_ax.legend(
            loc="lower left", bbox_to_anchor=(-0.15, -0.15), fontsize=7, facecolor=PALETTE.bg_alt, labelcolor=PALETTE.fg,
        )
        self._chart_canvas.draw()

    def get_target_name(self) -> str | None:
        """For AcquisitionPanel's "Target" tab header -- None until a
        search has actually resolved a target."""
        return self._target.name if self._target is not None else None

    def get_reference_name(self) -> str | None:
        """For AcquisitionPanel's "Reference star" tab header -- the
        target itself if it's already a standard (see is_standard_
        candidate), else whichever candidate is currently selected in the
        tree, else None if nothing's resolved yet."""
        if self._target is None:
            return None
        if is_standard_candidate(self._target):
            return self._target.name
        selection = self._tree.selection()
        if selection:
            return self._candidates[int(selection[0])].star.name
        return None

    def get_target_star(self) -> Star | None:
        """For AcquisitionPanel's "Target" tab GOTO button -- the full
        Star (RA/DEC included), not just its name."""
        return self._target

    def get_reference_star(self) -> Star | None:
        """Same selection logic as get_reference_name, but returns the
        full Star (RA/DEC included) for AcquisitionPanel's "Reference
        star" tab GOTO button."""
        if self._target is None:
            return None
        if is_standard_candidate(self._target):
            return self._target
        selection = self._tree.selection()
        if selection:
            return self._candidates[int(selection[0])].star
        return None

    def get_reference_spectrum(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Whatever's currently shown in the "Reference spectrum" plot
        above (real Pickles data if available, else the labeled blackbody
        fallback) -- reused as the source shape for the Reference star
        tab's live preview mock, so what that tab "shows arriving on the
        sensor" actually resembles the real star being observed instead
        of an unrelated placeholder curve."""
        return self._reference_spectrum

    def get_target_spectrum_model(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Best-guess continuum shape for the TARGET's own live preview
        mock. Unlike the reference star, the target's real spectrum is by
        definition unknown -- that's what the observation is for -- so
        this is never real archival data. If the target IS itself the
        reference (self-standard case, e.g. Vega), reuses that same real
        spectrum; otherwise falls back to a Teff-only blackbody model
        from its spectral type, same fallback role as catalog.
        model_spectrum elsewhere in this app. None if there's no target
        yet or its spectral type doesn't parse to a Teff estimate."""
        if self._target is None:
            return None
        if is_standard_candidate(self._target):
            return self._reference_spectrum
        teff = estimate_teff_k(self._target.spectral_type)
        if teff is None:
            return None
        wl, flux = model_spectrum(teff, wl_min=_ASSUMED_WL_MIN, wl_max=_ASSUMED_WL_MAX, n=900)
        return wl, _apply_reference_dips(wl, flux)


# -- Reference / Target acquisition tabs ---------------------------------------------------------------


class AcquisitionPanel(ttk.Frame):
    """Capture + calibration for ONE star (the standard reference, or the
    target) -- merged into a single tab rather than a separate global
    Calibration tab, because darks/offset frames must be taken at the
    SAME exposure/gain as the science frames they'll correct, and the
    reference and target often need different settings (a bright A0V
    standard vs. a fainter target) -- one shared Calibration tab couldn't
    reflect that; it also had no live histogram, which flats genuinely
    need (see FlatsPanel) but a plain dark/offset capture doesn't -- this
    panel still shows one since it's useful to confirm exposure/gain
    "looks right" before committing to a batch of darks.

    Flats are deliberately NOT here: they correct the optical train/
    sensor (vignetting, pixel response), not a specific star, so they
    only need doing once per setup rather than once per reference/target
    -- see FlatsPanel."""

    def __init__(
        self, parent: tk.Misc, role: str, seed: int, get_star, get_spectrum=None,
        mount_worker: MountWorker | None = None,
    ):
        super().__init__(parent, padding=10)
        self._role = role  # "reference" or "target"
        self._title = "Reference star" if role == "reference" else "Target"
        self._seed = seed
        self._get_star = get_star
        self._get_spectrum = get_spectrum
        self._mount_worker = mount_worker
        self._mount_connected = False
        self._goto_pending = False
        self._goto_target_ra_deg: float | None = None
        self._goto_target_dec_deg: float | None = None
        self._science_count = 0
        self._dark_count = 0
        self._offset_count = 0
        # Real captured frames (not just counters) -- what the Reduction
        # tab actually stacks/calibrates, see get_science_frames() etc.
        # below and spectro/reduction.py for the pipeline that consumes
        # them.
        self._science_frames: list[np.ndarray] = []
        self._dark_frames: list[np.ndarray] = []
        self._offset_frames: list[np.ndarray] = []
        self._capture_seed_counter = 0

        # Simulates imperfect GOTO pointing: the star doesn't land dead
        # center in the frame in real life, so ROI panning starts away
        # from it -- see _on_center_roi and _render_frame's status readout.
        pan_rng = np.random.default_rng(seed)
        self._true_pan_x = float(pan_rng.uniform(-150.0, 150.0))
        self._true_pan_y = float(pan_rng.uniform(-25.0, 25.0))

        self._header_var = tk.StringVar(value=f"{self._title}: (none selected yet)")
        ttk.Label(self, textvariable=self._header_var, font=("", 10, "bold")).pack(anchor="w")

        columns = ttk.Frame(self)
        columns.pack(fill="both", expand=True, pady=(10, 0))
        left = ttk.Frame(columns)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(columns)
        right.pack(side="left", fill="none", padx=(10, 0))

        roi_frame = ttk.LabelFrame(left, text="ROI framing -- find & center the star before capturing", padding=8)
        roi_frame.pack(fill="x")
        ttk.Label(roi_frame, text="Pan X").grid(row=0, column=0, sticky="w")
        self._roi_x_var = tk.DoubleVar(value=0.0)
        ttk.Scale(roi_frame, from_=-200, to=200, variable=self._roi_x_var, command=self._on_settings_changed).grid(
            row=0, column=1, sticky="we", padx=(8, 8),
        )
        ttk.Label(roi_frame, text="Pan Y").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._roi_y_var = tk.DoubleVar(value=0.0)
        ttk.Scale(roi_frame, from_=-40, to=40, variable=self._roi_y_var, command=self._on_settings_changed).grid(
            row=1, column=1, sticky="we", padx=(8, 8), pady=(4, 0),
        )
        roi_frame.columnconfigure(1, weight=1)
        self._roi_status_var = tk.StringVar(value="")
        self._roi_status_label = ttk.Label(roi_frame, textvariable=self._roi_status_var)
        self._roi_status_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(roi_frame, text="Center ROI (auto-detect)", command=self._on_center_roi).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0),
        )

        preview_frame = ttk.LabelFrame(left, text="Live preview -- order 0 (star) + order 1 (spectrum trail)", padding=8)
        preview_frame.pack(fill="both", expand=True, pady=(8, 0))
        self._preview_figure = Figure(figsize=(5.5, 1.5), dpi=100)
        self._preview_ax = self._preview_figure.add_subplot(111)
        self._preview_figure.patch.set_facecolor(PALETTE.bg)
        self._preview_canvas = FigureCanvasTkAgg(self._preview_figure, master=preview_frame)
        self._preview_canvas.get_tk_widget().pack(fill="both", expand=True)

        profile_frame = ttk.LabelFrame(left, text="Uncalibrated profile -- lines marked at their assumed position", padding=8)
        profile_frame.pack(fill="both", expand=True, pady=(8, 0))
        self._profile_figure = Figure(figsize=(5.5, 1.6), dpi=100)
        self._profile_ax = self._profile_figure.add_subplot(111)
        self._profile_canvas = FigureCanvasTkAgg(self._profile_figure, master=profile_frame)
        self._profile_canvas.get_tk_widget().pack(fill="both", expand=True)

        controls = ttk.LabelFrame(left, text="Exposure / gain", padding=8)
        controls.pack(fill="x", pady=(8, 0))
        ttk.Label(controls, text="Exposure").grid(row=0, column=0, sticky="w")
        self._exposure_var = tk.DoubleVar(value=35.0)
        ttk.Scale(controls, from_=0, to=100, variable=self._exposure_var, command=self._on_settings_changed).grid(
            row=0, column=1, sticky="we", padx=(8, 8),
        )
        self._exposure_label_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self._exposure_label_var, width=10).grid(row=0, column=2, sticky="w")
        ttk.Label(controls, text="Gain").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._gain_var = tk.DoubleVar(value=60.0)
        ttk.Scale(controls, from_=0, to=100, variable=self._gain_var, command=self._on_settings_changed).grid(
            row=1, column=1, sticky="we", padx=(8, 8), pady=(6, 0),
        )
        self._gain_label_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self._gain_label_var, width=10).grid(row=1, column=2, sticky="w", pady=(6, 0))
        controls.columnconfigure(1, weight=1)

        telescope_frame = ttk.LabelFrame(right, text="Telescope", padding=8)
        telescope_frame.pack(fill="x")
        self._goto_status_var = tk.StringVar(value="No star selected yet.")
        ttk.Label(
            telescope_frame, textvariable=self._goto_status_var, foreground=PALETTE.fg_dim, wraplength=200, justify="left",
        ).pack(anchor="w")
        ttk.Button(telescope_frame, text="GOTO", command=self._on_goto).pack(anchor="w", pady=(6, 0))

        status_frame = ttk.LabelFrame(right, text="This session", padding=8)
        status_frame.pack(fill="x", pady=(10, 0))
        self._science_var = tk.StringVar(value=f"⬜ {self._title} spectrum (0 frames)")
        ttk.Label(status_frame, textvariable=self._science_var, wraplength=200, justify="left").pack(anchor="w", pady=2)
        self._dark_var = tk.StringVar(value="⬜ Darks (0 frames)")
        ttk.Label(status_frame, textvariable=self._dark_var, width=26).pack(anchor="w", pady=2)
        self._offset_var = tk.StringVar(value="⬜ Offset/bias (0 frames)")
        ttk.Label(status_frame, textvariable=self._offset_var, width=26).pack(anchor="w", pady=2)
        ttk.Label(
            status_frame, foreground=PALETTE.fg_dim, wraplength=200, justify="left",
            text="Flats are shared, not per-star -- see the Flats tab.",
        ).pack(anchor="w", pady=(6, 0))

        frames_row = ttk.Frame(right)
        frames_row.pack(anchor="w", pady=(10, 0))
        ttk.Label(frames_row, text="Frames to stack").pack(side="left")
        self._science_frames_var = tk.IntVar(value=20)
        ttk.Spinbox(frames_row, from_=1, to=200, textvariable=self._science_frames_var, width=5).pack(
            side="left", padx=(6, 0),
        )
        ttk.Button(right, text=f"Capture {self._title.lower()}", command=self._on_capture_science).pack(
            anchor="w", pady=(6, 0),
        )
        self._dark_hint_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self._dark_hint_var, foreground=PALETTE.fg_dim, wraplength=190, justify="left").pack(
            anchor="w", pady=(10, 2),
        )
        ttk.Button(right, text="Capture darks (same settings)", command=self._on_capture_dark).pack(anchor="w")
        self._offset_hint_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self._offset_hint_var, foreground=PALETTE.fg_dim, wraplength=190, justify="left").pack(
            anchor="w", pady=(10, 2),
        )
        ttk.Button(right, text="Capture offset/bias (min exposure)", command=self._on_capture_offset).pack(anchor="w")

        self._stats_var = tk.StringVar(value="fps: --   frames: --")
        ttk.Label(right, textvariable=self._stats_var, foreground=PALETTE.fg_dim).pack(anchor="w", pady=(10, 0))

        # Small and out of the way -- the uncalibrated profile above is
        # the primary "is this actually working" readout for a star; this
        # is just here for a quick over/under-exposure sanity check.
        ttk.Label(right, text="Histogram", foreground=PALETTE.fg_dim).pack(anchor="w", pady=(10, 0))
        self._hist_figure = Figure(figsize=(2.3, 1.0), dpi=100)
        self._hist_ax = self._hist_figure.add_subplot(111)
        self._hist_canvas = FigureCanvasTkAgg(self._hist_figure, master=right)
        self._hist_canvas.get_tk_widget().pack(anchor="w")

        self._live_frame_count = 0
        self._on_settings_changed()
        self._refresh_header()
        self._live_tick()

    def _refresh_header(self) -> None:
        star = self._get_star()
        name = star.name if star is not None else None
        self._header_var.set(f"{self._title}: {name}" if name else f"{self._title}: (none selected yet -- pick one in Target & standard)")
        self.after(500, self._refresh_header)

    def _on_goto(self) -> None:
        if self._mount_worker is None:
            return
        if not self._mount_connected:
            self._goto_status_var.set("Mount not connected -- connect it in the Connection tab first.")
            return
        star = self._get_star()
        if star is None:
            self._goto_status_var.set("No star selected yet -- pick one in Target & standard.")
            return
        self._goto_pending = True
        self._goto_target_ra_deg = star.ra_deg
        self._goto_target_dec_deg = star.dec_deg
        self._goto_status_var.set(f"Slewing to {star.name}...")
        self._mount_worker.goto(star.ra_deg / 15.0, star.dec_deg)

    def handle_mount_event(self, event: WorkerEvent) -> None:
        """Fed by App._pump_events -- every AcquisitionPanel sees every
        mount event (there's only one mount), but only reacts to
        goto_result/position while ITS OWN GOTO is the one pending, so
        the Reference star tab doesn't show the Target tab's slew status
        or vice versa."""
        if event.kind == "connected":
            self._mount_connected = True
        elif event.kind == "disconnected":
            self._mount_connected = False
            self._goto_pending = False
        elif event.kind == "goto_result" and self._goto_pending:
            self._goto_status_var.set(f"GOTO: {event.payload['meaning']}")
            if event.payload.get("code") != 0:
                self._goto_pending = False  # rejected outright (e.g. below horizon) -- nothing more will arrive
        elif event.kind == "position" and self._goto_pending and self._goto_target_ra_deg is not None:
            current_ra_deg = event.payload["ra_hours"] * 15.0
            current_dec_deg = event.payload["dec_deg"]
            separation_deg = angular_separation_deg(
                self._goto_target_ra_deg, self._goto_target_dec_deg, current_ra_deg, current_dec_deg,
            )
            if separation_deg < 0.05:  # ~3 arcmin -- plenty tight for "close enough to see it drift into frame"
                self._goto_status_var.set(f"Arrived -- within {separation_deg * 3600:.0f}\" of target")
                self._goto_pending = False
            else:
                self._goto_status_var.set(f"Slewing... {separation_deg:.2f}° from target")

    def _exposure_ms(self) -> float:
        return 10.0 + self._exposure_var.get() * 20.0  # 10-2010ms -- display/model only, not wired to a real camera

    def _gain_value(self) -> int:
        return round(self._gain_var.get() * 5.7)  # 0-570, matches this project's mock camera range

    def _on_settings_changed(self, _value: str | None = None) -> None:
        self._exposure_label_var.set(format_exposure_us(self._exposure_ms() * 1000.0))
        self._gain_label_var.set(str(self._gain_value()))
        self._dark_hint_var.set(f"Matches current settings: {self._exposure_label_var.get()}, gain {self._gain_label_var.get()}")
        self._offset_hint_var.set(f"Minimum exposure, gain {self._gain_label_var.get()} -- exposure length doesn't matter for bias")
        self._render_frame(frame_seed=None)

    def _on_center_roi(self) -> None:
        self._roi_x_var.set(self._true_pan_x)
        self._roi_y_var.set(self._true_pan_y)
        self._on_settings_changed()

    def _live_tick(self) -> None:
        # Redraws with a fresh noise/jitter draw every _LIVE_INTERVAL_MS,
        # independent of any slider move -- see _synthetic_trail_image's
        # docstring for why this is what actually reads as "live" rather
        # than a plot that only ever changes when you touch a control.
        if self.winfo_ismapped():
            self._live_frame_count += 1
            self._render_frame(frame_seed=self._seed * 100_003 + self._live_frame_count)
            self._stats_var.set(f"fps: {1000.0 / _LIVE_INTERVAL_MS:.1f}   frames: {self._live_frame_count}")
        self.after(_LIVE_INTERVAL_MS, self._live_tick)

    def _current_capture_params(self) -> tuple[float, tuple[np.ndarray, np.ndarray] | None, float, float]:
        """(brightness, spectrum, roi_offset_x, roi_offset_y) for whatever
        the exposure/gain/ROI controls are set to RIGHT NOW -- the single
        place both the live preview (_render_frame) and the actual frame
        capture (_on_capture_science/_dark/_offset) read these from, so a
        captured frame always matches what the operator was just looking
        at when they clicked the button."""
        brightness = 0.4 + 1.2 * (self._exposure_var.get() / 100.0) * (0.5 + self._gain_var.get() / 100.0)
        spectrum = self._get_spectrum() if self._get_spectrum is not None else None
        # How far the star still is from where the trail/lines are assumed
        # to sit (see _synthetic_trail_image) -- nonzero until the operator
        # pans the ROI to match the star's actual (imperfect-GOTO) position.
        offset_x = self._true_pan_x - self._roi_x_var.get()
        offset_y = self._true_pan_y - self._roi_y_var.get()
        return brightness, spectrum, offset_x, offset_y

    def _next_capture_seed(self) -> int:
        self._capture_seed_counter += 1
        return self._seed * 1_000_003 + self._capture_seed_counter

    def _render_frame(self, frame_seed: int | None) -> None:
        brightness, spectrum, offset_x, offset_y = self._current_capture_params()
        image = _synthetic_trail_image(
            seed=self._seed, brightness_scale=brightness, spectrum=spectrum, frame_seed=frame_seed,
            roi_offset_x=offset_x, roi_offset_y=offset_y,
        )
        centered = abs(offset_x) < _ROI_TOLERANCE_X and abs(offset_y) < _ROI_TOLERANCE_Y
        if centered:
            self._roi_status_var.set("Star centered -- ready to capture")
            self._roi_status_label.configure(foreground=PALETTE.accent_ok)
        else:
            self._roi_status_var.set("Star not centered -- pan ROI to find it (or auto-center)")
            self._roi_status_label.configure(foreground=PALETTE.accent_warn)
        self._preview_ax.clear()
        self._preview_ax.imshow(image, cmap="inferno", aspect="auto")
        _draw_extraction_band(self._preview_ax, image.shape[1])
        _draw_center_guide(self._preview_ax, centered)
        # The overlay patches above extend autoscale margins past the
        # image's own extent (visible as a blank strip of the axes'
        # default background past the right/top edge) -- pin the view
        # back to exactly the image bounds now that everything's drawn.
        self._preview_ax.set_xlim(-0.5, image.shape[1] - 0.5)
        self._preview_ax.set_ylim(image.shape[0] - 0.5, -0.5)
        self._preview_ax.set_xticks([])
        self._preview_ax.set_yticks([])
        for spine in self._preview_ax.spines.values():
            spine.set_visible(False)
        self._preview_canvas.draw()
        _draw_profile(self._profile_ax, self._profile_figure, image)
        self._profile_canvas.draw()
        _draw_histogram(self._hist_ax, self._hist_figure, image, compact=True)
        self._hist_canvas.draw()

    def _on_capture_science(self) -> None:
        # Multiple REAL frames per click, not one -- each with its own
        # independent noise draw (a fresh frame_seed) so stacking them
        # later (see spectro/reduction.py's stack_frames) actually
        # improves SNR over a single frame, same reason darks/offset are
        # already batched.
        n = max(1, self._science_frames_var.get())
        brightness, spectrum, offset_x, offset_y = self._current_capture_params()
        for _ in range(n):
            frame = _synthetic_trail_image(
                seed=self._seed, brightness_scale=brightness, spectrum=spectrum,
                frame_seed=self._next_capture_seed(), roi_offset_x=offset_x, roi_offset_y=offset_y,
            )
            self._science_frames.append(frame)
        self._science_count += n
        self._science_var.set(f"✅ {self._title} spectrum ({self._science_count} frames)")

    def _on_capture_dark(self) -> None:
        # No star signal at all (cap on) -- brightness_scale is
        # irrelevant here since include_signal=False skips the only
        # things it would have scaled.
        for _ in range(20):
            frame = _synthetic_trail_image(seed=self._seed, frame_seed=self._next_capture_seed(), include_signal=False)
            self._dark_frames.append(frame)
        self._dark_count += 20
        self._dark_var.set(f"✅ Darks ({self._dark_count} frames)")

    def _on_capture_offset(self) -> None:
        for _ in range(20):
            frame = _synthetic_trail_image(seed=self._seed, frame_seed=self._next_capture_seed(), include_signal=False)
            self._offset_frames.append(frame)
        self._offset_count += 20
        self._offset_var.set(f"✅ Offset/bias ({self._offset_count} frames)")

    def get_science_frames(self) -> list[np.ndarray]:
        return self._science_frames

    def get_dark_frames(self) -> list[np.ndarray]:
        return self._dark_frames

    def get_offset_frames(self) -> list[np.ndarray]:
        return self._offset_frames


# -- Flats tab ---------------------------------------------------------------


class FlatsPanel(ttk.Frame):
    """Flat frames correct the optical train + sensor (vignetting, pixel-
    to-pixel sensitivity, dust) -- NOT tied to a specific star, so unlike
    darks/offset (see AcquisitionPanel) these only need doing once per
    setup, not once per reference/target. A good flat needs the exposure
    tuned so the histogram peak sits at ~2/3 of full well -- doing that
    from a plain "Capture 20" button with no live feedback (the previous
    single Calibration tab) isn't actually usable in practice, hence the
    live preview + histogram here."""

    def __init__(self, parent: tk.Misc):
        super().__init__(parent, padding=10)
        self._flat_count = 0
        self._flat_frames: list[np.ndarray] = []
        self._master_flat: np.ndarray | None = None
        self._capture_seed_counter = 0

        ttk.Label(
            self, foreground=PALETTE.fg_dim, wraplength=800, justify="left",
            text=(
                "Even illumination (twilight sky / flat panel / diffuser over the aperture). Tune exposure "
                "until the histogram peak sits at the 2/3 line, then capture."
            ),
        ).pack(anchor="w")

        columns = ttk.Frame(self)
        columns.pack(fill="both", expand=True, pady=(10, 0))
        left = ttk.Frame(columns)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(columns)
        right.pack(side="left", fill="none", padx=(10, 0))

        preview_frame = ttk.LabelFrame(left, text="Live preview -- flat field", padding=8)
        preview_frame.pack(fill="both", expand=True)
        self._preview_figure = Figure(figsize=(5.5, 1.9), dpi=100)
        self._preview_ax = self._preview_figure.add_subplot(111)
        self._preview_figure.patch.set_facecolor(PALETTE.bg)
        self._preview_canvas = FigureCanvasTkAgg(self._preview_figure, master=preview_frame)
        self._preview_canvas.get_tk_widget().pack(fill="both", expand=True)

        hist_frame = ttk.LabelFrame(left, text="Live histogram -- aim for the 2/3 line", padding=8)
        hist_frame.pack(fill="both", expand=True, pady=(8, 0))
        self._hist_figure = Figure(figsize=(5.5, 1.5), dpi=100)
        self._hist_ax = self._hist_figure.add_subplot(111)
        self._hist_canvas = FigureCanvasTkAgg(self._hist_figure, master=hist_frame)
        self._hist_canvas.get_tk_widget().pack(fill="both", expand=True)

        controls = ttk.LabelFrame(left, text="Exposure", padding=8)
        controls.pack(fill="x", pady=(8, 0))
        ttk.Label(controls, text="Exposure").grid(row=0, column=0, sticky="w")
        self._exposure_var = tk.DoubleVar(value=45.0)
        ttk.Scale(controls, from_=0, to=100, variable=self._exposure_var, command=self._on_settings_changed).grid(
            row=0, column=1, sticky="we", padx=(8, 8),
        )
        self._level_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self._level_var, width=16).grid(row=0, column=2, sticky="w")
        controls.columnconfigure(1, weight=1)

        status_frame = ttk.LabelFrame(right, text="This session", padding=8)
        status_frame.pack(fill="x")
        self._flat_var = tk.StringVar(value="⬜ Flats (0 frames)")
        ttk.Label(status_frame, textvariable=self._flat_var, width=24).pack(anchor="w", pady=2)

        ttk.Button(right, text="Capture flats", command=self._on_capture).pack(anchor="w", pady=(10, 0))
        ttk.Button(right, text="Build master flat", command=self._on_build_master).pack(anchor="w", pady=(4, 0))
        self._master_var = tk.StringVar(value="No master flat built yet.")
        ttk.Label(right, textvariable=self._master_var, foreground=PALETTE.fg_dim, wraplength=190, justify="left").pack(
            anchor="w", pady=(10, 0),
        )

        self._live_frame_count = 0
        self._on_settings_changed()
        self._live_tick()

    def _on_settings_changed(self, _value: str | None = None) -> None:
        self._render_frame(frame_seed=None)

    def _live_tick(self) -> None:
        # Same self-paced idiom as AcquisitionPanel._live_tick -- see its
        # docstring for why a periodic redraw (not just on slider moves)
        # is what actually reads as a live feed.
        if self.winfo_ismapped():
            self._live_frame_count += 1
            self._render_frame(frame_seed=3 * 100_003 + self._live_frame_count)
        self.after(_LIVE_INTERVAL_MS, self._live_tick)

    def _render_frame(self, frame_seed: int | None) -> None:
        brightness_pct = self._exposure_var.get()
        image = _synthetic_flat_image(seed=3, brightness_pct=brightness_pct, frame_seed=frame_seed)
        peak_pct = float(image.mean()) / 255.0 * 100.0
        note = "-- good" if 60.0 <= peak_pct <= 73.0 else ("-- too dim" if peak_pct < 60.0 else "-- too bright")
        self._level_var.set(f"{peak_pct:.0f}% {note}")
        self._preview_ax.clear()
        self._preview_ax.imshow(image, cmap="gray", aspect="auto", vmin=0, vmax=255)
        self._preview_ax.set_xticks([])
        self._preview_ax.set_yticks([])
        for spine in self._preview_ax.spines.values():
            spine.set_visible(False)
        self._preview_canvas.draw()
        _draw_histogram(self._hist_ax, self._hist_figure, image)
        self._hist_canvas.draw()

    def _on_capture(self) -> None:
        brightness_pct = self._exposure_var.get()
        for _ in range(20):
            self._capture_seed_counter += 1
            frame = _synthetic_flat_image(
                seed=3, brightness_pct=brightness_pct, frame_seed=3 * 1_000_003 + self._capture_seed_counter,
            )
            self._flat_frames.append(frame)
        self._flat_count += 20
        self._flat_var.set(f"✅ Flats ({self._flat_count} frames)")

    def _on_build_master(self) -> None:
        if not self._flat_frames:
            self._master_var.set("Capture some flats first.")
            return
        self._master_flat = stack_frames(self._flat_frames)
        self._master_var.set(f"Master flat built from {self._flat_count} frames.")

    def get_flat_frames(self) -> list[np.ndarray]:
        return self._flat_frames

    def get_master_flat(self) -> np.ndarray | None:
        return self._master_flat


# -- Reduction tab ---------------------------------------------------------------


class ReductionPanel(ttk.Frame):
    """The real signal-processing pipeline (spectro/reduction.py) applied
    to whatever's actually been captured in the Reference star, Target,
    and Flats tabs -- four sequential steps, each needing the previous
    one's output:

    1. Build master calibration frames (dark/offset per star, stacked).
    2. Reduce + stack the reference star (dark+flat correction, then a
       real line-DETECTION pass -- not just trusting the assumed pixel
       positions -- to fit an actual dispersion).
    3. Reduce + stack the target the same way (no line detection here:
       the target's real lines are exactly what this whole observation
       is trying to discover, so nothing to detect them against yet).
    4. Derive the instrument's response from the reference (comparing its
       calibrated profile to its REAL known spectrum, see TargetPanel.
       get_reference_spectrum) and apply that correction to the target --
       the actual flux calibration step.

    Every number here comes from the synthetic-but-realistic frames the
    other tabs generated -- this module doesn't know or care that the
    frames are synthetic; the same code would run unchanged on real
    captures, which is the whole point of building it for real instead of
    a canned demo plot."""

    def __init__(
        self, parent: tk.Misc, reference_panel: AcquisitionPanel, target_capture_panel: AcquisitionPanel,
        flats_panel: FlatsPanel, target_panel: TargetPanel,
    ):
        super().__init__(parent, padding=10)
        self._reference_panel = reference_panel
        self._target_capture_panel = target_capture_panel
        self._flats_panel = flats_panel
        self._target_panel = target_panel

        self._reference_master_dark: np.ndarray | None = None
        self._reference_master_offset: np.ndarray | None = None
        self._target_master_dark: np.ndarray | None = None
        self._target_master_offset: np.ndarray | None = None
        self._reference_calibrated: np.ndarray | None = None
        self._target_calibrated: np.ndarray | None = None
        self._dispersion: tuple[float, float] | None = None
        self._final_wl: np.ndarray | None = None
        self._final_flux: np.ndarray | None = None
        self._result_version = 0

        columns = ttk.Frame(self)
        columns.pack(fill="both", expand=True)
        left = ttk.Frame(columns)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(columns)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self._step1_status_var, self._step1_status_label, *_unused1 = self._build_step(
            left, "1. Build master calibration frames (dark/offset, per star)",
            "Build masters", self._on_build_masters, with_plot=False,
        )
        (
            self._step2_status_var, self._step2_status_label,
            self._step2_figure, self._step2_ax, self._step2_canvas,
        ) = self._build_step(
            left, "2. Reduce + stack -- Reference star", "Reduce reference", self._on_reduce_reference, with_plot=True,
        )
        (
            self._step3_status_var, self._step3_status_label,
            self._step3_figure, self._step3_ax, self._step3_canvas,
        ) = self._build_step(
            right, "3. Reduce + stack -- Target", "Reduce target", self._on_reduce_target, with_plot=True,
        )
        (
            self._step4_status_var, self._step4_status_label,
            self._step4_figure, self._step4_ax, self._step4_canvas,
        ) = self._build_step(
            right, "4. Flux calibration (reference -> target)", "Compute & apply", self._on_calibrate_flux, with_plot=True,
        )

    def _build_step(self, parent: tk.Misc, title: str, button_text: str, command, with_plot: bool) -> tuple:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill="both" if with_plot else "x", expand=with_plot, pady=(0, 10))
        ttk.Button(frame, text=button_text, command=command).pack(anchor="w")
        status_var = tk.StringVar(value="")
        status_label = ttk.Label(frame, textvariable=status_var, foreground=PALETTE.fg_dim, wraplength=480, justify="left")
        status_label.pack(anchor="w", pady=(6, 0))
        if not with_plot:
            return status_var, status_label, None, None, None
        figure = Figure(figsize=(5, 1.3), dpi=100)
        ax = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=(6, 0))
        style_axes(figure, ax)
        canvas.draw()
        return status_var, status_label, figure, ax, canvas

    def _set_status(self, label: ttk.Label, var: tk.StringVar, message: str, ok: bool) -> None:
        var.set(message)
        label.configure(foreground=PALETTE.accent_ok if ok else PALETTE.accent_warn)

    def _on_build_masters(self) -> None:
        lines = []
        ok = True
        for label, panel in (("Reference", self._reference_panel), ("Target", self._target_capture_panel)):
            darks = panel.get_dark_frames()
            offsets = panel.get_offset_frames()
            if not darks or not offsets:
                lines.append(f"{label}: capture darks AND offset/bias frames first.")
                ok = False
                continue
            master_dark = stack_frames(darks)
            master_offset = stack_frames(offsets)
            if panel is self._reference_panel:
                self._reference_master_dark, self._reference_master_offset = master_dark, master_offset
            else:
                self._target_master_dark, self._target_master_offset = master_dark, master_offset
            lines.append(
                f"{label}: {len(darks)} darks + {len(offsets)} offset frames stacked "
                f"(bias level ~{master_offset.mean():.1f} ADU).",
            )
        master_flat = self._flats_panel.get_master_flat()
        if master_flat is None:
            lines.append("Flats: build a master flat in the Flats tab first.")
            ok = False
        else:
            lines.append(f"Flats: master flat ready (mean {master_flat.mean():.1f} ADU).")
        self._set_status(self._step1_status_label, self._step1_status_var, "\n".join(lines), ok)

    def _plot_profile_with_lines(self, ax, figure, canvas, profile: np.ndarray, detected: list) -> None:
        ax.clear()
        pixels = np.arange(len(profile))
        ax.plot(
            pixels[_TRAIL_START_PX:_TRAIL_END_PX], profile[_TRAIL_START_PX:_TRAIL_END_PX],
            color=PALETTE.accent, linewidth=1,
        )
        detected_px = {label: px for label, _wl, px in detected}
        for label, wl in REFERENCE_LINES:
            assumed_px = _assumed_px_for_wavelength(wl)
            if _TRAIL_START_PX <= assumed_px <= _TRAIL_END_PX:
                ax.axvline(assumed_px, color=PALETTE.fg_dim, linewidth=0.6, linestyle=":")
            if label in detected_px:
                px = detected_px[label]
                ax.axvline(px, color=PALETTE.accent_ok, linewidth=0.9)
                ax.text(
                    px, 1.02, label, color=PALETTE.accent_ok, fontsize=6, ha="center", va="bottom",
                    transform=ax.get_xaxis_transform(),
                )
        ax.set_xlabel("pixel (dotted = assumed, solid = detected)", fontsize=8)
        ax.set_ylabel("counts (calibrated)", fontsize=8)
        style_axes(figure, ax)
        canvas.draw()

    def _on_reduce_reference(self) -> None:
        if self._reference_master_dark is None or self._reference_master_offset is None:
            self._set_status(self._step2_status_label, self._step2_status_var, "Run step 1 (build masters) first.", False)
            return
        science = self._reference_panel.get_science_frames()
        master_flat = self._flats_panel.get_master_flat()
        if not science:
            self._set_status(self._step2_status_label, self._step2_status_var, "Capture reference science frames first.", False)
            return
        if master_flat is None:
            self._set_status(self._step2_status_label, self._step2_status_var, "Build a master flat in the Flats tab first.", False)
            return
        stack = stack_frames(science)
        try:
            calibrated = calibrate_science(stack, self._reference_master_dark, master_flat, self._reference_master_offset)
        except ReductionError as exc:
            self._set_status(self._step2_status_label, self._step2_status_var, f"Reduction failed: {exc}", False)
            return
        self._reference_calibrated = calibrated
        profile = extract_profile(calibrated)
        detected = detect_line_pixels(profile)
        self._dispersion = fit_dispersion(detected)
        gain = snr_gain(len(science))
        if self._dispersion is not None:
            a, _b = self._dispersion
            msg = (
                f"{len(science)} frames stacked (SNR x{gain:.1f}). Detected {len(detected)}/{len(REFERENCE_LINES)} "
                f"reference lines -- fitted dispersion {a:.3f} Å/px."
            )
        else:
            msg = (
                f"{len(science)} frames stacked (SNR x{gain:.1f}). Only {len(detected)} line(s) detected -- "
                "need at least 2 to fit a real dispersion, falling back to the assumed one."
            )
        self._set_status(self._step2_status_label, self._step2_status_var, msg, True)
        self._plot_profile_with_lines(self._step2_ax, self._step2_figure, self._step2_canvas, profile, detected)

    def _on_reduce_target(self) -> None:
        if self._target_master_dark is None or self._target_master_offset is None:
            self._set_status(self._step3_status_label, self._step3_status_var, "Run step 1 (build masters) first.", False)
            return
        science = self._target_capture_panel.get_science_frames()
        master_flat = self._flats_panel.get_master_flat()
        if not science:
            self._set_status(self._step3_status_label, self._step3_status_var, "Capture target science frames first.", False)
            return
        if master_flat is None:
            self._set_status(self._step3_status_label, self._step3_status_var, "Build a master flat in the Flats tab first.", False)
            return
        stack = stack_frames(science)
        try:
            calibrated = calibrate_science(stack, self._target_master_dark, master_flat, self._target_master_offset)
        except ReductionError as exc:
            self._set_status(self._step3_status_label, self._step3_status_var, f"Reduction failed: {exc}", False)
            return
        self._target_calibrated = calibrated
        profile = extract_profile(calibrated)
        gain = snr_gain(len(science))
        msg = f"{len(science)} frames stacked (SNR x{gain:.1f}). Target's own lines are unknown -- nothing to detect yet."
        self._set_status(self._step3_status_label, self._step3_status_var, msg, True)
        self._plot_profile_with_lines(self._step3_ax, self._step3_figure, self._step3_canvas, profile, [])

    def _on_calibrate_flux(self) -> None:
        if self._reference_calibrated is None:
            self._set_status(self._step4_status_label, self._step4_status_var, "Run step 2 (reduce reference) first.", False)
            return
        if self._target_calibrated is None:
            self._set_status(self._step4_status_label, self._step4_status_var, "Run step 3 (reduce target) first.", False)
            return
        reference_spectrum = self._target_panel.get_reference_spectrum()
        if reference_spectrum is None:
            self._set_status(
                self._step4_status_label, self._step4_status_var,
                "No reference spectrum available -- pick a target/standard in the Target & standard tab first.", False,
            )
            return
        ref_wl, ref_flux = reference_spectrum
        _pixels, _wavelengths, response, _measured = compute_response(
            self._reference_calibrated, ref_wl, ref_flux, self._dispersion,
        )
        final_wl, final_flux = apply_response(self._target_calibrated, response, self._dispersion)
        self._final_wl, self._final_flux = final_wl, final_flux
        self._result_version += 1
        dispersion_note = (
            f"real fitted dispersion ({self._dispersion[0]:.3f} Å/px)" if self._dispersion is not None
            else "assumed linear dispersion (not enough lines detected in step 2 to refit)"
        )
        reference_name = self._target_panel.get_reference_name() or "reference"
        target_name = self._target_panel.get_target_name() or "target"
        msg = f"Response derived from {reference_name}, applied to {target_name} -- using the {dispersion_note}."
        self._set_status(self._step4_status_label, self._step4_status_var, msg, True)
        self._step4_ax.clear()
        self._step4_ax.plot(final_wl, final_flux, color=PALETTE.accent_ok, linewidth=1)
        for label, line_wl in REFERENCE_LINES:
            if final_wl.min() < line_wl < final_wl.max():
                self._step4_ax.axvline(line_wl, color=PALETTE.border, linewidth=0.8)
                self._step4_ax.text(line_wl, 1.02, label, color=PALETTE.fg_dim, fontsize=6, ha="center", va="bottom", transform=self._step4_ax.get_xaxis_transform())
        self._step4_ax.set_xlabel("wavelength (Å)", fontsize=8)
        self._step4_ax.set_ylabel("flux (norm.)", fontsize=8)
        style_axes(self._step4_figure, self._step4_ax)
        self._step4_canvas.draw()

    def get_final_spectrum(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._final_wl is None or self._final_flux is None:
            return None
        return self._final_wl, self._final_flux

    def get_result_version(self) -> int:
        """Increments every time get_final_spectrum's result changes --
        SpectrumPanel polls this (cheap int compare) instead of the
        arrays themselves to know when to redraw."""
        return self._result_version

    def get_dispersion(self) -> tuple[float, float] | None:
        return self._dispersion


# -- Calibration / spectrum tab ---------------------------------------------------------------


class SpectrumPanel(ttk.Frame):
    """Final result tab -- shows the REAL calibrated spectrum computed by
    ReductionPanel's 4-step pipeline (spectro/reduction.py), not a canned
    demo plot. Polls ReductionPanel's result_version periodically (same
    self.after() polling idiom as TargetPanel's _poll_results) since the
    pipeline's output changes whenever the operator (re-)runs one of its
    steps, and there's no event/callback wiring between tabs elsewhere in
    this app either."""

    def __init__(
        self, parent: tk.Misc, reduction_panel: ReductionPanel, target_panel: TargetPanel,
        connection_panel: ConnectionPanel,
    ):
        super().__init__(parent, padding=10)
        self._reduction_panel = reduction_panel
        self._target_panel = target_panel
        self._connection_panel = connection_panel
        self._last_result_version = -1

        self._header_var = tk.StringVar(
            value="No result yet -- run the Reduction tab's 4 steps, then come back here.",
        )
        ttk.Label(self, textvariable=self._header_var, font=("", 10, "bold"), wraplength=1000, justify="left").pack(
            anchor="w",
        )

        result_frame = ttk.LabelFrame(self, text="Calibrated, response-corrected spectrum", padding=8)
        result_frame.pack(fill="both", expand=True, pady=(10, 0))
        self._figure = Figure(figsize=(9, 2.0), dpi=100)
        self._ax = self._figure.add_subplot(111)
        style_axes(self._figure, self._ax)
        self._canvas = FigureCanvasTkAgg(self._figure, master=result_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas.draw()

        self._stats_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._stats_var, foreground=PALETTE.fg_dim).pack(anchor="w", pady=(6, 0))

        export_row = ttk.Frame(self)
        export_row.pack(fill="x", pady=(10, 0))
        ttk.Button(export_row, text="Export FITS", command=self._on_export_fits).pack(side="left")
        ttk.Button(export_row, text="Export CSV", command=self._on_export_csv).pack(side="left", padx=(6, 0))
        self._export_status_var = tk.StringVar(value="")
        ttk.Label(export_row, textvariable=self._export_status_var, foreground=PALETTE.fg_dim).pack(
            side="left", padx=(16, 0),
        )

        avspec_frame = ttk.LabelFrame(self, text="Export for AAVSO AVSpec submission", padding=8)
        avspec_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(
            avspec_frame, foreground=PALETTE.fg_dim, wraplength=1000, justify="left",
            text=(
                "Fills in the FITS header fields AVSpec expects (site, instrument, observer code, "
                "wavelength axis) -- you still submit it yourself at aavso.org/apps/avspec/submit, "
                "and it still goes through AAVSO's own validation. Refused while Mock is selected in "
                "the Connection tab, so a practice run can't be passed off as a real observation."
            ),
        ).pack(anchor="w")
        avspec_row = ttk.Frame(avspec_frame)
        avspec_row.pack(fill="x", pady=(6, 0))
        ttk.Label(avspec_row, text="Obscode").grid(row=0, column=0, sticky="w")
        self._observer_code_var = tk.StringVar(value="")
        ttk.Entry(avspec_row, textvariable=self._observer_code_var, width=10).grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Label(avspec_row, text="Site name").grid(row=0, column=2, sticky="w")
        self._site_name_var = tk.StringVar(value="")
        ttk.Entry(avspec_row, textvariable=self._site_name_var, width=18).grid(row=0, column=3, sticky="w", padx=(4, 12))
        ttk.Label(avspec_row, text="Instrument name").grid(row=0, column=4, sticky="w")
        self._instrument_name_var = tk.StringVar(value="")
        ttk.Entry(avspec_row, textvariable=self._instrument_name_var, width=18).grid(row=0, column=5, sticky="w", padx=(4, 0))
        ttk.Label(
            avspec_frame, foreground=PALETTE.fg_dim,
            text="Site name and instrument name must match exactly what's registered at app.aavso.org/site_equip/.",
        ).pack(anchor="w", pady=(4, 0))
        ttk.Button(avspec_frame, text="Export for AVSpec...", command=self._on_export_avspec).pack(anchor="w", pady=(6, 0))
        self._avspec_status_var = tk.StringVar(value="")
        ttk.Label(avspec_frame, textvariable=self._avspec_status_var, foreground=PALETTE.fg_dim, wraplength=1000, justify="left").pack(
            anchor="w", pady=(4, 0),
        )

        self.after(500, self._poll)

    def _poll(self) -> None:
        version = self._reduction_panel.get_result_version()
        if version != self._last_result_version:
            self._last_result_version = version
            spectrum = self._reduction_panel.get_final_spectrum()
            if spectrum is not None:
                self._render(spectrum)
        self.after(500, self._poll)

    def _render(self, spectrum: tuple[np.ndarray, np.ndarray]) -> None:
        wl, flux = spectrum
        target_name = self._target_panel.get_target_name() or "target"
        reference_name = self._target_panel.get_reference_name() or "reference"
        self._header_var.set(f"{target_name} -- flux-calibrated against {reference_name}")

        self._ax.clear()
        self._ax.plot(wl, flux, color=PALETTE.accent_ok, linewidth=1)
        for label, line_wl in REFERENCE_LINES:
            if wl.min() < line_wl < wl.max():
                idx = int(np.argmin(np.abs(wl - line_wl)))
                if not np.isfinite(flux[idx]):
                    continue  # outside the reference spectrum's own calibrated range (see apply_response) -- no valid flux to label here
                self._ax.axvline(line_wl, color=PALETTE.border, linewidth=0.8)
                self._ax.text(line_wl, flux[idx] + 0.05, label, color=PALETTE.fg_dim, fontsize=7, ha="center")
        self._ax.set_xlabel("wavelength (Å)")
        self._ax.set_ylabel("relative flux")
        style_axes(self._figure, self._ax)
        self._canvas.draw()

        dispersion = self._reduction_panel.get_dispersion()
        if dispersion is not None:
            a, _b = dispersion
            r_at_halpha = resolution_at(6562.79, a)
            self._stats_var.set(f"Dispersion: {a:.3f} Å/px (fitted from detected lines)   R ≈ {r_at_halpha:.0f} @ Hα")
        else:
            self._stats_var.set("Dispersion: assumed linear mapping -- not enough lines were detected in step 2 to fit a real one.")

    def _on_export_csv(self) -> None:
        spectrum = self._reduction_panel.get_final_spectrum()
        if spectrum is None:
            self._export_status_var.set("Nothing to export yet -- run the reduction pipeline first.")
            return
        wl, flux = spectrum
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="Export calibrated spectrum",
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wavelength_angstrom", "relative_flux"])
            writer.writerows(zip(wl.tolist(), flux.tolist(), strict=True))
        self._export_status_var.set(f"Exported {len(wl)} points to {path}")

    def _on_export_fits(self) -> None:
        spectrum = self._reduction_panel.get_final_spectrum()
        if spectrum is None:
            self._export_status_var.set("Nothing to export yet -- run the reduction pipeline first.")
            return
        wl, flux = spectrum
        path = filedialog.asksaveasfilename(
            defaultextension=".fits", filetypes=[("FITS", "*.fits")], title="Export calibrated spectrum",
        )
        if not path:
            return
        columns = fits.ColDefs([
            fits.Column(name="wavelength", format="D", unit="Angstrom", array=wl),
            fits.Column(name="flux", format="D", array=flux),
        ])
        fits.BinTableHDU.from_columns(columns).writeto(path, overwrite=True)
        self._export_status_var.set(f"Exported {len(wl)} points to {path}")

    def _on_export_avspec(self) -> None:
        if self._connection_panel.is_mock():
            self._avspec_status_var.set(
                "Refused: Mock is selected in the Connection tab. Connect real hardware before "
                "exporting for AVSpec submission -- see AcquisitionPanel's own docstring for what's "
                "real vs. mock in this app.",
            )
            return
        spectrum = self._reduction_panel.get_final_spectrum()
        if spectrum is None:
            self._avspec_status_var.set("Nothing to export yet -- run the reduction pipeline first.")
            return
        target_name = self._target_panel.get_target_name()
        if target_name is None:
            self._avspec_status_var.set("No target star resolved yet -- pick one in Target & standard.")
            return
        observer_code = self._observer_code_var.get().strip()
        site_name = self._site_name_var.get().strip()
        instrument_name = self._instrument_name_var.get().strip()
        if not observer_code or not site_name or not instrument_name:
            self._avspec_status_var.set("Fill in Obscode, site name, and instrument name first.")
            return
        try:
            site_lat_deg = self._connection_panel.get_site_lat_deg()
            site_lon_deg = self._connection_panel.get_site_lon_deg()
            site_elevation_m = self._connection_panel.get_site_elevation_m()
        except ValueError:
            self._avspec_status_var.set("Invalid site lat/lon/elevation in the Connection tab.")
            return
        wl, flux = spectrum
        path = filedialog.asksaveasfilename(
            defaultextension=".fits", filetypes=[("FITS", "*.fits")], title="Export for AVSpec submission",
        )
        if not path:
            return
        try:
            write_avspec_fits(
                path, wl, flux, object_name=target_name, observer_code=observer_code, site_name=site_name,
                site_lat_deg=site_lat_deg, site_lon_deg=site_lon_deg, site_elevation_m=site_elevation_m,
                instrument_name=instrument_name,
            )
        except AvspecExportError as exc:
            self._avspec_status_var.set(f"Export failed: {exc}")
            return
        self._avspec_status_var.set(
            f"Exported {len(wl)} points to {path} -- submit it yourself at aavso.org/apps/avspec/submit.",
        )
