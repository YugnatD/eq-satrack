"""Frontend for the Star Analyser spectroscopy app.

TargetPanel is wired to a real star catalog (spectro/catalog.py, SIMBAD via
astroquery) -- target search and standard-star candidates are real lookups,
not demo data. Mount AND camera CONNECTION are real (ConnectionPanel owns a
real am5.gui.worker.MountWorker and a real camera.worker.CameraWorker,
unchanged from the ISS tracker -- same devices, same code). What's still
synthetic is FRAME ACQUISITION: AcquisitionPanel/FlatsPanel/AlignmentPanel
generate synthetic-but-realistic frames rather than reading live frames off
a connected real camera -- wiring that up needs a physical ASI290MC to test
against (the same reason the mount protocol itself was only trusted after
characterize.py runs against real hardware, not written blind), so it's
deliberately deferred rather than guessed at. ReductionPanel and
SpectrumPanel run the REAL reduction pipeline (spectro/reduction.py:
stacking, dark/flat calibration, line detection, dispersion fitting,
instrument-response flux calibration) on those frames regardless -- the
same code would run unchanged on real captures once that wiring exists.
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
from camera.worker import CameraEvent, CameraWorker
from spectro.alignment import angle_from_points, extract_aligned_crop, paste_star_trail
from spectro.gui.live_camera import LiveCameraFeed, RealCaptureState, TabResyncTracker
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
from spectro.session import Session
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

# Where the star is assumed to sit within a properly-identified local
# working patch (_synthetic_trail_image's own fixed nominal anchor), and
# the fixed shape spectro/reduction.py's own geometry constants (and
# spectro.alignment.extract_aligned_crop's default local_anchor) assume --
# single source of truth so nothing here can silently drift apart.
_ROI_TARGET_X, _ROI_TARGET_Y = 55, _TRAIL_ROW

# Typical amateur GOTO pointing accuracy, in arcmin -- used to simulate a
# realistic initial pointing error (see AcquisitionPanel's _true_pan_x/y),
# converted to pixels via ConnectionPanel.get_plate_scale_arcsec_per_px
# (a real focal-length-dependent formula) instead of an arbitrary pixel
# range. Roughly isotropic -- real GOTO error isn't meaningfully worse in
# one axis than the other, so the same value is used for both.
_TYPICAL_GOTO_ACCURACY_ARCMIN = 3.0

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
    frame_seed: int | None = None, include_signal: bool = True, dispersion_a: float | None = None,
    angle_deg: float = 0.0,
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

    `include_signal=False` skips the star blob AND the trail entirely --
    what a real DARK or OFFSET/BIAS frame looks like (cap on, no optical
    signal at all, only the sensor's own bias/read/shot noise) -- used by
    AcquisitionPanel's dark/offset capture instead of reusing a science
    frame with the star just left in.

    `dispersion_a`, if given, is the real Å/px predicted from the
    operator's actual grating + optical setup (see ConnectionPanel.
    get_dispersion_a_per_px) -- painted onto the trail instead of the
    fixed placeholder rate, so the mock's line spacing actually reflects
    the chosen Star Analyser + distance.

    `angle_deg` tilts the trail direction away from the image's own +x
    axis, around the fixed nominal anchor (55, h/2) -- 0 reproduces the
    original "properly framed AND perfectly horizontal" image every
    other function here assumes (TRAIL_START_PX etc.); this function
    always returns that same "already aligned" local coordinate system
    regardless of angle_deg, panning/positioning within a larger sensor
    is a separate concern handled by spectro.alignment.paste_star_trail,
    not by this function -- see AcquisitionPanel's full-frame view for
    where the two get composed."""
    noise_rng = np.random.default_rng(frame_seed if frame_seed is not None else seed)
    h, w = 90, 420
    bias, read_noise = 8.0, 3.0
    img = noise_rng.normal(bias, read_noise, size=(h, w))

    if include_signal:
        order0_x = 55 + noise_rng.normal(0.0, 0.6)  # seeing/tracking jitter, a pixel or so
        order0_y = h / 2.0 + noise_rng.normal(0.0, 0.5)
        yy, xx = np.mgrid[0:h, 0:w]
        img += 220.0 * brightness_scale * np.exp(-(((xx - order0_x) ** 2 + (yy - order0_y) ** 2) / (2 * 3.5**2)))

        wl, flux = spectrum if spectrum is not None else _synthetic_spectrum(seed)
        # Distance along/across the (possibly tilted) dispersion axis,
        # measured from the FIXED nominal anchor (55, h/2) -- not the
        # jittery blob position above, so the trail doesn't visibly
        # wobble frame to frame the way the star's own seeing jitter does.
        theta = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx, dy = xx - 55.0, yy - (h / 2.0)
        along = dx * cos_t + dy * sin_t
        perp = -dx * sin_t + dy * cos_t
        px_along = 55.0 + along
        trail_wl = _wavelength_for_px(px_along, dispersion_a)
        trail_flux = np.interp(trail_wl.ravel(), wl, flux, left=0.0, right=0.0).reshape(xx.shape)
        img += np.where(
            px_along >= _TRAIL_START_PX,
            trail_flux * 150.0 * brightness_scale * np.exp(-(perp**2) / (2 * 4.0**2)),
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


def _synthetic_full_frame(
    seed: int, sensor_width: int, sensor_height: int, star_xy: tuple[float, float], angle_deg: float,
    spectrum: tuple[np.ndarray, np.ndarray] | None = None, brightness_scale: float = 1.0,
    frame_seed: int | None = None, dispersion_a: float | None = None,
) -> np.ndarray:
    """Full-sensor-sized mock frame -- what the operator actually sees in
    a real live view before identifying anything, unlike
    _synthetic_trail_image's small "already found and framed" patch.
    Generates that same small patch (unchanged, still fast) and pastes it
    into a full `(sensor_height, sensor_width)` canvas at `star_xy` (real
    sensor pixel coordinates) and `angle_deg`, via spectro.alignment.
    paste_star_trail -- ~200ms at a real 1936x1096 ASI290MC size. Too slow
    for the fast (140ms) tick the small local patches use, so
    AcquisitionPanel only calls this on-demand (button/GOTO triggered);
    AlignmentPanel does tick it, but on its own much slower dedicated
    interval (see its _LIVE_INTERVAL_MS) since it's a practice view, not
    something exposure/gain tuning depends on being high-fps."""
    background_rng = np.random.default_rng(frame_seed if frame_seed is not None else seed)
    canvas = background_rng.normal(8.0, 3.0, size=(sensor_height, sensor_width))
    patch = _synthetic_trail_image(
        seed=seed, brightness_scale=brightness_scale, spectrum=spectrum, frame_seed=frame_seed,
        dispersion_a=dispersion_a, angle_deg=angle_deg,
    )
    return np.clip(paste_star_trail(canvas, patch, star_xy, angle_deg), 0, 255)


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


def _draw_profile(ax, figure, image: np.ndarray, dispersion_a: float | None = None) -> None:
    """Uncalibrated profile (raw pixel counts vs. pixel position) with the
    real reference lines marked at their ASSUMED pixel position (see
    _assumed_px_for_wavelength, using the real grating+distance dispersion
    if dispersion_a is given) -- NOT a calibrated spectrum, just a live
    sanity check that the trail actually looks like a stellar spectrum
    (dips roughly where the Balmer series etc. should be) while framing/
    focusing, well before the real wavelength calibration step in the
    Spectrum tab."""
    ax.clear()
    profile = _extract_profile(image)
    pixels = np.arange(len(profile))
    ax.plot(pixels[_TRAIL_START_PX:], profile[_TRAIL_START_PX:], color=PALETTE.accent, linewidth=1)
    for label, wl in REFERENCE_LINES:
        px = _assumed_px_for_wavelength(wl, dispersion_a)
        if _TRAIL_START_PX <= px <= _TRAIL_END_PX:
            ax.axvline(px, color=PALETTE.border, linewidth=0.8)
            ax.text(
                px, 1.02, label, color=PALETTE.fg_dim, fontsize=6, ha="center", va="bottom",
                transform=ax.get_xaxis_transform(),
            )
    ax.set_xlabel("pixel (assumed dispersion, not yet calibrated)", fontsize=8)
    ax.set_ylabel("counts", fontsize=8)
    style_axes(figure, ax)


def _draw_extraction_band(ax, w: int) -> None:
    """Outlines exactly which rows get summed into the extracted profile
    (see _extract_profile) -- without this, it's not obvious the profile
    plot only reads a thin band around the trail rather than the whole
    frame."""
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


# -- Connection tab ---------------------------------------------------------------


class ConnectionPanel(ttk.Frame):
    """Mount AND camera connection are both REAL -- this panel owns a real
    am5.gui.worker.MountWorker and a real camera.worker.CameraWorker
    (unchanged from the ISS tracker), talking to either mock devices or
    actual serial/USB hardware exactly the way that project's own
    ConnectionPanel does. Manual jog itself lives in a separate floating
    window (spectro/gui/jog_window.py, owned by App, same shown-not-
    destroyed pattern as the ISS tracker's own JogWindow) rather than
    embedded here, so it's reachable from any tab -- see
    on_connection_change, which App wires to that window's
    set_connected(). Connecting the camera here only opens the device
    handle -- AcquisitionPanel/FlatsPanel/AlignmentPanel still generate
    synthetic frames rather than reading live ones off it (see this
    module's own docstring for why that's deliberately deferred)."""

    # Real Star Analyser groove densities -- keys double as the Grating
    # combobox's own values, so there's one source of truth for both.
    _GRATING_LINES_PER_MM = {
        "Star Analyser SA-100 (100 l/mm)": 100.0,
        "Star Analyser SA-200 (200 l/mm)": 200.0,
    }

    def __init__(
        self, parent: tk.Misc, mount_worker: MountWorker, camera_worker: CameraWorker, on_connection_change=None,
        live_camera_feed: LiveCameraFeed | None = None,
    ):
        super().__init__(parent, padding=10)
        self._mount_worker = mount_worker
        self._camera_worker = camera_worker
        self._live_camera_feed = live_camera_feed
        self._on_connection_change = on_connection_change
        self._mount_connected = False
        self._camera_connected = False
        columns = ttk.Frame(self)
        columns.pack(fill="both", expand=True)
        left = ttk.Frame(columns)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(columns)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self._build_mount_control(left)
        self._build_camera_control(left)

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
        self._grating_var = tk.StringVar(value="Star Analyser SA-100 (100 l/mm)")
        ttk.Combobox(
            grating_frame, textvariable=self._grating_var, state="readonly", width=28,
            values=list(self._GRATING_LINES_PER_MM),
        ).grid(row=0, column=1, sticky="w")
        self._grating_var.trace_add("write", lambda *_a: self._update_dispersion_label())
        ttk.Label(grating_frame, text="Focal length (mm)").grid(row=1, column=0, sticky="w", pady=(4, 0))
        # NOT part of get_dispersion_a_per_px -- that's driven by the
        # grating-to-sensor distance below, not the telescope's focal
        # length (confirmed against the Paton Hawksley Star Analyser
        # manual's own formula). Focal length DOES drive plate scale/FOV
        # below, and the simulated GOTO pointing error in AcquisitionPanel
        # (see _true_pan_x/y there).
        self._focal_length_var = tk.StringVar(value="1000")
        ttk.Entry(grating_frame, textvariable=self._focal_length_var, width=10).grid(
            row=1, column=1, sticky="w", pady=(4, 0),
        )
        self._focal_length_var.trace_add("write", lambda *_a: self._update_dispersion_label())

        ttk.Label(grating_frame, text="Grating to sensor distance (mm)").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self._grating_distance_var = tk.StringVar(value="30")
        distance_entry = ttk.Entry(grating_frame, textvariable=self._grating_distance_var, width=10)
        distance_entry.grid(row=2, column=1, sticky="w", pady=(4, 0))
        self._grating_distance_var.trace_add("write", lambda *_a: self._update_dispersion_label())

        ttk.Label(grating_frame, text="Camera pixel size (µm)").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self._pixel_size_var = tk.StringVar(value="2.9")
        pixel_size_entry = ttk.Entry(grating_frame, textvariable=self._pixel_size_var, width=10)
        pixel_size_entry.grid(row=3, column=1, sticky="w", pady=(4, 0))
        self._pixel_size_var.trace_add("write", lambda *_a: self._update_dispersion_label())
        ttk.Label(
            grating_frame, text="2.9 by default (ASI290MC/MM) -- change if using a different camera or binning.",
            foreground=PALETTE.fg_dim, wraplength=280, justify="left",
        ).grid(row=4, column=0, columnspan=2, sticky="w")

        ttk.Label(grating_frame, text="Sensor width x height (px)").grid(row=5, column=0, sticky="w", pady=(4, 0))
        sensor_row = ttk.Frame(grating_frame)
        sensor_row.grid(row=5, column=1, sticky="w", pady=(4, 0))
        self._sensor_width_var = tk.StringVar(value="1936")
        self._sensor_height_var = tk.StringVar(value="1096")
        ttk.Entry(sensor_row, textvariable=self._sensor_width_var, width=6).pack(side="left")
        ttk.Label(sensor_row, text="x").pack(side="left", padx=2)
        ttk.Entry(sensor_row, textvariable=self._sensor_height_var, width=6).pack(side="left")
        self._sensor_width_var.trace_add("write", lambda *_a: self._update_dispersion_label())
        self._sensor_height_var.trace_add("write", lambda *_a: self._update_dispersion_label())

        self._dispersion_label_var = tk.StringVar(value="")
        ttk.Label(grating_frame, textvariable=self._dispersion_label_var, foreground=PALETTE.fg_dim).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(4, 0),
        )
        self._plate_scale_label_var = tk.StringVar(value="")
        ttk.Label(grating_frame, textvariable=self._plate_scale_label_var, foreground=PALETTE.fg_dim).grid(
            row=7, column=0, columnspan=2, sticky="w",
        )
        ttk.Label(
            grating_frame, foreground=PALETTE.fg_dim, wraplength=280, justify="left",
            text=(
                "Dispersion drives line markers/search windows and the live "
                "preview mock (Paton Hawksley Star Analyser manual's formula). "
                "Plate scale/FOV drive the simulated GOTO pointing error in "
                "the Reference star/Target tabs' ROI framing."
            ),
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self._update_dispersion_label()

    @staticmethod
    def _set_radios_locked(radios: list[ttk.Radiobutton], locked: bool) -> None:
        """Shared by the mount and camera Mock/Real(/Serial/TCP) radio
        groups -- locked from the moment Connect is clicked (see
        _on_mount_connect/_on_camera_connect) until disconnected or a
        connect_error, so the kind var can't be flipped out from under an
        in-flight or completed connection (see is_mock's own docstring
        for why that would matter)."""
        state = "disabled" if locked else "normal"
        for radio in radios:
            radio.configure(state=state)

    def _build_mount_control(self, parent: tk.Misc) -> None:
        frame = ttk.LabelFrame(parent, text="Mount (AM3/AM5)", padding=8)
        frame.pack(fill="x")

        self._mount_kind_var = tk.StringVar(value="mock")
        self._mount_address_var = tk.StringVar(value="/dev/ttyACM0")
        self._mount_seed_var = tk.StringVar(value="")
        self._mount_kind_radios: list[ttk.Radiobutton] = []
        for i, (label, value) in enumerate((("Mock", "mock"), ("Serial", "serial"), ("TCP", "tcp"))):
            radio = ttk.Radiobutton(
                frame, text=label, variable=self._mount_kind_var, value=value,
                command=self._update_mount_address_state,
            )
            radio.grid(row=0, column=i, sticky="w")
            self._mount_kind_radios.append(radio)
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
        # Locked from the click, not just once "connected" arrives -- a
        # real serial/TCP connect can take a moment, and flipping the
        # kind var during that window would have is_mock() read the NEW
        # kind once "connected" fires even though the device that
        # actually opened is whatever kind was selected at click time.
        # Re-enabled on connect_error below (connection never happened)
        # or on "disconnected" (see handle_mount_event).
        self._set_radios_locked(self._mount_kind_radios, True)
        self._mount_status_var.set("Connecting...")
        self._mount_worker.connect(
            self._mount_kind_var.get(), address=self._mount_address_var.get(),
            mock_seed=int(seed_text) if seed_text else None,
            latitude_deg=latitude_deg, longitude_deg=longitude_deg,
        )

    def _build_camera_control(self, parent: tk.Misc) -> None:
        frame = ttk.LabelFrame(parent, text="Camera (ASI290MC + Star Analyser)", padding=8)
        frame.pack(fill="x", pady=(10, 0))

        self._camera_kind_var = tk.StringVar(value="mock")
        self._camera_id_var = tk.StringVar(value="0")
        self._camera_kind_radios: list[ttk.Radiobutton] = [
            ttk.Radiobutton(frame, text="Mock", variable=self._camera_kind_var, value="mock"),
            ttk.Radiobutton(frame, text="Real ASI camera", variable=self._camera_kind_var, value="real"),
        ]
        self._camera_kind_radios[0].grid(row=0, column=0, sticky="w")
        self._camera_kind_radios[1].grid(row=0, column=1, sticky="w")
        ttk.Label(frame, text="camera id", foreground=PALETTE.fg_dim).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(frame, textvariable=self._camera_id_var, width=6).grid(row=1, column=1, sticky="w", pady=(4, 0))

        self._camera_connect_button = ttk.Button(frame, text="Connect", command=self._on_camera_connect)
        self._camera_connect_button.grid(row=2, column=0, pady=(8, 0))
        self._camera_disconnect_button = ttk.Button(
            frame, text="Disconnect", command=self._camera_worker.disconnect, state="disabled",
        )
        self._camera_disconnect_button.grid(row=2, column=1, pady=(8, 0))
        self._camera_status_var = tk.StringVar(value="Not connected")
        ttk.Label(frame, textvariable=self._camera_status_var, foreground=PALETTE.fg_dim).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(6, 0),
        )
        ttk.Label(
            frame, foreground=PALETTE.fg_dim, wraplength=280, justify="left",
            text=(
                "Real: Reference star/Target/Flats read live frames off this camera. Mock: they still "
                "generate synthetic frames (see this module's docstring)."
            ),
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _on_camera_connect(self) -> None:
        try:
            camera_id = int(self._camera_id_var.get())
        except ValueError:
            self._camera_status_var.set("Invalid camera id")
            return
        self._camera_connect_button.configure(state="disabled")
        # Locked from the click -- same race-window reasoning as the
        # mount's own copy of this (see _on_mount_connect).
        self._set_radios_locked(self._camera_kind_radios, True)
        self._camera_status_var.set("Connecting...")
        kind = self._camera_kind_var.get()
        # Set BEFORE issuing connect -- the "connected" event payload
        # itself carries no kind info (CameraWorker's mock backend
        # connects exactly like a real one), so this is the only place
        # that knows which was actually requested. See LiveCameraFeed.
        # is_active, which reads this instead of `connected` alone.
        if self._live_camera_feed is not None:
            self._live_camera_feed.kind = kind
        self._camera_worker.connect(
            kind, camera_id=camera_id, plate_scale_arcsec_per_px=self.get_plate_scale_arcsec_per_px(),
        )

    def handle_camera_event(self, event: CameraEvent) -> None:
        if event.kind == "connected":
            self._camera_connected = True
            colour = "colour" if event.payload["is_color"] else "mono"
            self._camera_status_var.set(
                f"Connected -- {event.payload['width']}x{event.payload['height']} {colour}, "
                f"{event.payload.get('bit_depth', 8)}-bit",
            )
            # Already disabled from the click (see _on_camera_connect) --
            # reasserted here too in case anything ever re-enables them
            # out of band.
            self._set_radios_locked(self._camera_kind_radios, True)
            self._camera_disconnect_button.configure(state="normal")
        elif event.kind == "connect_error":
            self._camera_status_var.set(f"Connection failed: {event.payload['message']}")
            self._camera_connect_button.configure(state="normal")
            # Connection never actually happened -- undo the click-time
            # lock so a different kind can be picked and retried.
            self._set_radios_locked(self._camera_kind_radios, False)
        elif event.kind == "disconnected":
            self._camera_connected = False
            self._camera_status_var.set("Not connected")
            self._camera_connect_button.configure(state="normal")
            self._camera_disconnect_button.configure(state="disabled")
            self._set_radios_locked(self._camera_kind_radios, False)

    def get_site_lat_deg(self) -> float:
        return float(self._site_lat_var.get())

    def get_site_lon_deg(self) -> float:
        return float(self._site_lon_var.get())

    def get_site_elevation_m(self) -> float:
        return float(self._site_elevation_var.get())

    def get_dispersion_a_per_px(self) -> float | None:
        """Predicted Å/pixel for the currently selected Grating, Grating-
        to-sensor distance, and camera pixel size -- real formula from the
        Paton Hawksley Star Analyser 100 manual (Wavelength Calibration
        section): dispersion(Å/px) = 10000 * pixel_size(um) /
        [grating_lines_per_mm * grating_to_sensor_distance(mm)]. Feeds
        spectro/reduction.py's REAL pipeline (line-search windows,
        response calibration fallback), not just the live-preview mock --
        pixel size is a field, not a hardcoded camera assumption, since
        that pipeline is meant to run unchanged on real captured frames
        from whatever camera/binning is actually in use. None if the
        distance or pixel size fields are empty/invalid/non-positive --
        see spectro.reduction.assumed_dispersion for the placeholder
        fallback used everywhere this feeds into."""
        lines_per_mm = self._GRATING_LINES_PER_MM.get(self._grating_var.get())
        try:
            distance_mm = float(self._grating_distance_var.get())
            pixel_size_um = float(self._pixel_size_var.get())
        except ValueError:
            return None
        if lines_per_mm is None or distance_mm <= 0 or pixel_size_um <= 0:
            return None
        return 10000.0 * pixel_size_um / (lines_per_mm * distance_mm)

    def get_plate_scale_arcsec_per_px(self) -> float | None:
        """Standard plate-scale formula: 206265 * pixel_size(mm) /
        focal_length(mm) -- how many arcsec of sky one pixel covers.
        Unrelated to spectral dispersion (that's the grating+distance
        formula above); this is what actually depends on the telescope's
        focal length. None if focal length or pixel size are empty/
        invalid/non-positive."""
        try:
            focal_length_mm = float(self._focal_length_var.get())
            pixel_size_um = float(self._pixel_size_var.get())
        except ValueError:
            return None
        if focal_length_mm <= 0 or pixel_size_um <= 0:
            return None
        return 206265.0 * (pixel_size_um / 1000.0) / focal_length_mm

    def get_fov_arcmin(self) -> tuple[float, float] | None:
        """(width, height) field of view in arcmin, from plate scale and
        sensor resolution. None if plate scale or sensor dimensions
        aren't available/valid."""
        plate_scale = self.get_plate_scale_arcsec_per_px()
        dimensions = self.get_sensor_dimensions()
        if plate_scale is None or dimensions is None:
            return None
        width_px, height_px = dimensions
        return plate_scale * width_px / 60.0, plate_scale * height_px / 60.0

    def get_sensor_dimensions(self) -> tuple[int, int] | None:
        """(width, height) in pixels, from the Sensor width x height
        field -- the single source of truth for how big the full-frame
        mock/live view is (AcquisitionPanel, AlignmentPanel), not just
        FOV math, so a changed sensor size is consistent everywhere.
        None if either field is empty/invalid/non-positive."""
        try:
            width_px = int(float(self._sensor_width_var.get()))
            height_px = int(float(self._sensor_height_var.get()))
        except ValueError:
            return None
        if width_px <= 0 or height_px <= 0:
            return None
        return width_px, height_px

    def get_instrument_metadata(self) -> dict:
        """Everything about the current optical/site/device setup worth
        recording per session -- see Session.write_metadata, called once
        from ReductionPanel._on_build_masters."""
        return {
            "grating": self._grating_var.get(),
            "focal_length_mm": self._focal_length_var.get(),
            "grating_to_sensor_distance_mm": self._grating_distance_var.get(),
            "pixel_size_um": self._pixel_size_var.get(),
            "sensor_dimensions_px": self.get_sensor_dimensions(),
            "dispersion_a_per_px": self.get_dispersion_a_per_px(),
            "plate_scale_arcsec_per_px": self.get_plate_scale_arcsec_per_px(),
            "site_lat_deg": self._site_lat_var.get(),
            "site_lon_deg": self._site_lon_var.get(),
            "site_elevation_m": self._site_elevation_var.get(),
            "mount_kind": self._mount_kind_var.get(),
            "camera_kind": self._camera_kind_var.get(),
        }

    def _update_dispersion_label(self) -> None:
        dispersion_a = self.get_dispersion_a_per_px()
        if dispersion_a is None:
            self._dispersion_label_var.set("Predicted dispersion: -- (enter a valid distance)")
        else:
            self._dispersion_label_var.set(f"Predicted dispersion: {dispersion_a:.2f} Å/px")

        plate_scale = self.get_plate_scale_arcsec_per_px()
        fov = self.get_fov_arcmin()
        if plate_scale is None or fov is None:
            self._plate_scale_label_var.set("Plate scale: -- (enter a valid focal length)")
        else:
            self._plate_scale_label_var.set(f"Plate scale: {plate_scale:.2f} arcsec/px   FOV: {fov[0]:.1f} x {fov[1]:.1f} arcmin")

    def is_mock(self) -> bool:
        """True unless BOTH mount and camera are set to Real AND actually,
        currently connected -- used to refuse exports meant for real
        submission (see SpectrumPanel's AVSpec export), so a synthetic run
        can't produce a file that looks like a real observation. Requires
        actual CONNECTION, not just the radio button selection, now that
        AcquisitionPanel/FlatsPanel genuinely read live frames from the
        camera once it's connected (see LiveCameraFeed/RealCaptureState)
        -- selecting Real but not yet connecting must still count as mock.
        Also checks the camera's own kind (not just "is something
        connected"): CameraWorker's mock backend connects successfully
        too (see _on_camera_connect), which must NOT count as real."""
        mount_is_real = self._mount_kind_var.get() != "mock" and self._mount_connected
        camera_is_real = self._camera_kind_var.get() == "real" and self._camera_connected
        return not (mount_is_real and camera_is_real)

    def handle_mount_event(self, event: WorkerEvent) -> None:
        if event.kind == "connected":
            self._mount_connected = True
            self._mount_status_var.set(f"Connected -- firmware {event.payload['firmware']}")
            self._mount_disconnect_button.configure(state="normal")
            # Already disabled from the moment Connect was clicked (see
            # _on_mount_connect) -- reasserted here too in case anything
            # ever re-enables them out of band.
            self._set_radios_locked(self._mount_kind_radios, True)
            if self._on_connection_change is not None:
                self._on_connection_change(True)
        elif event.kind == "connect_error":
            self._mount_status_var.set(f"Connection failed: {event.payload['message']}")
            self._mount_connect_button.configure(state="normal")
            # Connection never actually happened -- undo the click-time
            # lock (see _on_mount_connect) so a different kind can be
            # picked and retried.
            self._set_radios_locked(self._mount_kind_radios, False)
        elif event.kind == "disconnected":
            self._mount_connected = False
            self._mount_status_var.set("Not connected")
            self._mount_connect_button.configure(state="normal")
            self._mount_disconnect_button.configure(state="disabled")
            self._set_radios_locked(self._mount_kind_radios, False)
            if self._on_connection_change is not None:
                self._on_connection_change(False)


# -- Shared full-frame scroll-to-zoom -----------------------------------------------------
#
# Used by both AlignmentPanel and AcquisitionPanel's full-frame views --
# the same "click to mark a real sensor-pixel coordinate" interaction
# benefits from the same zoom in both places, and the math (zoom around
# the cursor, clamped so the view can't pan off the sensor) doesn't
# depend on anything panel-specific.


def _clamp_span(lo: float, hi: float, bound_lo: float, bound_hi: float) -> tuple[float, float]:
    """Shifts (lo, hi) -- a span already known to fit within bound_hi -
    bound_lo -- so both ends land inside [bound_lo, bound_hi], without
    changing its width. Keeps a zoomed view from panning off the edge of
    the sensor."""
    span = hi - lo
    if lo < bound_lo:
        lo, hi = bound_lo, bound_lo + span
    if hi > bound_hi:
        hi, lo = bound_hi, bound_hi - span
    return lo, hi


def _zoomed_view(
    event: object, cur_xlim: tuple[float, float], cur_ylim: tuple[float, float], width: int, height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """New (xlim, ylim) after a scroll-wheel zoom step around the cursor,
    or None if `event` isn't a real in-axes scroll (nothing to do). Zooms
    in/out around (event.xdata, event.ydata) -- data coordinates, which
    are always real sensor pixels regardless of the current zoom (see
    each caller's own imshow extent=), so marking stays accurate at any
    zoom level. Clamped via _clamp_span so the view never exceeds or
    pans outside the full (0, width) x (0, height) sensor extent."""
    if event.xdata is None or event.ydata is None:  # type: ignore[attr-defined]
        return None
    if event.button == "up":  # type: ignore[attr-defined]
        factor = 0.8  # scroll up -- zoom in, standard map-style convention
    elif event.button == "down":  # type: ignore[attr-defined]
        factor = 1.25  # zoom out
    else:
        return None
    x0, x1 = cur_xlim
    y_bottom, y_top = cur_ylim  # bottom > top -- matches extent=(0,w,h,0)
    span_x = x1 - x0
    span_y = y_bottom - y_top
    new_span_x = min(float(width), max(width * 0.02, span_x * factor))
    new_span_y = min(float(height), max(height * 0.02, span_y * factor))
    fx = (event.xdata - x0) / span_x  # type: ignore[attr-defined]
    fy = (event.ydata - y_top) / span_y  # type: ignore[attr-defined]
    new_x0 = event.xdata - fx * new_span_x  # type: ignore[attr-defined]
    new_y_top = event.ydata - fy * new_span_y  # type: ignore[attr-defined]
    new_x0, new_x1 = _clamp_span(new_x0, new_x0 + new_span_x, 0.0, float(width))
    new_y_top, new_y_bottom = _clamp_span(new_y_top, new_y_top + new_span_y, 0.0, float(height))
    return (new_x0, new_x1), (new_y_bottom, new_y_top)


# -- Alignment tab ---------------------------------------------------------------


class AlignmentPanel(ttk.Frame):
    """Once per session -- not once per star, see AcquisitionPanel's own
    docstring for that split -- find out how far the Star Analyser's
    dispersion axis actually is from the sensor's own horizontal, by
    looking at a full raw frame and tracing it by hand: mark order 0,
    then click-drag-release a line along the trail. This angle is a
    property of the physical rig (however the grating happens to sit in
    its nosepiece), not of any particular target, so it's measured here
    once and reused by every AcquisitionPanel tab -- the same "shared,
    not per-star" split this app already uses for flats vs. darks.

    The demo star/trail shown here always has SOME small random tilt
    (a fixed seed, not zero) so there's always something real to trace --
    a perfectly horizontal demo would defeat the entire point."""

    _DEMO_SEED = 101
    _DEMO_ANGLE_RANGE_DEG = 8.0
    _LARGE_ANGLE_WARNING_DEG = 5.0
    # Much slower than the module-level _LIVE_INTERVAL_MS (140ms) the
    # small local patches elsewhere tick at -- _synthetic_full_frame costs
    # ~200ms at a real 1936x1096 size (see its own docstring), too slow
    # for that cadence. This view is just a practice demo, not something
    # exposure/gain tuning depends on being high-fps, so a slower dedicated
    # tick is fine and still reads as a live feed rather than a photo.
    _LIVE_INTERVAL_MS = 800

    def __init__(self, parent: tk.Misc, connection_panel: ConnectionPanel, live_camera_feed: LiveCameraFeed | None = None):
        super().__init__(parent, padding=10)
        self._connection_panel = connection_panel
        self._live_camera_feed = live_camera_feed
        self._mode: str | None = None  # None, "order0", or "trace"
        self._order0_xy: tuple[float, float] | None = None
        self._trace_points: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._drag_start: tuple[float, float] | None = None
        self._measured_angle_deg: float | None = None
        self._demo_star_xy = (0.0, 0.0)
        self._demo_true_angle_deg = 0.0
        self._sensor_width = 1936
        self._sensor_height = 1096
        self._last_frame: np.ndarray | None = None
        self._live_frame_count = 0
        # None means "full sensor extent" -- see _redraw_current, which
        # fills these in the first time a frame is shown so _on_scroll
        # always has a real window to zoom from.
        self._view_xlim: tuple[float, float] | None = None
        self._view_ylim: tuple[float, float] | None = None
        # Tracks the demo/real edge so _render_frame can clear stale
        # marks/angle/zoom exactly once on the transition -- see its own
        # comment for why a demo-measured angle must not silently survive
        # into real captures.
        self._live_mode_was_real = False

        ttk.Label(
            self, foreground=PALETTE.fg_dim, wraplength=1000, justify="left",
            text=(
                "Do this once per session (not once per star) -- it measures how the Star Analyser is "
                "physically rotated relative to the sensor, which doesn't change between targets unless "
                "you touch the camera/grating orientation. 1) Click \"Mark order 0\", then click the star "
                "in the frame below. 2) Click \"Draw trace\", then click-drag-release along the spectrum. "
                "The measured angle then feeds every Reference star/Target tab. Scroll the mouse wheel over "
                "the frame to zoom in/out around the cursor for more precise marking -- this only changes "
                "the view, not what gets captured."
            ),
        ).pack(anchor="w")

        self._mode_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._mode_var, foreground=PALETTE.fg_dim).pack(anchor="w", pady=(4, 0))

        button_row = ttk.Frame(self)
        button_row.pack(fill="x", pady=(8, 0))
        self._order0_button = ttk.Button(button_row, text="Mark order 0", command=self._on_toggle_order0_mode)
        self._order0_button.pack(side="left")
        self._trace_button = ttk.Button(button_row, text="Draw trace", command=self._on_toggle_trace_mode)
        self._trace_button.pack(side="left", padx=(6, 0))
        ttk.Button(button_row, text="Reset", command=self._on_reset).pack(side="left", padx=(6, 0))
        self._new_demo_button = ttk.Button(button_row, text="New demo frame", command=self._on_new_demo_frame)
        self._new_demo_button.pack(side="left", padx=(6, 0))

        frame_box = ttk.LabelFrame(self, text="Full-frame live view", padding=8)
        frame_box.pack(fill="both", expand=True, pady=(8, 0))
        self._figure = Figure(figsize=(9, 3.3), dpi=100)
        self._figure.patch.set_facecolor(PALETTE.bg)
        self._ax = self._figure.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._figure, master=frame_box)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self._canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)

        self._status_var = tk.StringVar(value="")
        self._status_label = ttk.Label(self, textvariable=self._status_var, wraplength=1000, justify="left")
        self._status_label.pack(anchor="w", pady=(8, 0))

        self._on_new_demo_frame()
        self._live_tick()

    def _is_real(self) -> bool:
        return self._live_camera_feed is not None and self._live_camera_feed.is_active

    def _on_new_demo_frame(self) -> None:
        if self._is_real():
            return  # no "demo" concept once a real camera is live -- button is disabled too, see _update_status
        rng = np.random.default_rng()  # genuinely random each click -- a fresh practice tilt, not reproducible on purpose
        dims = self._connection_panel.get_sensor_dimensions() if self._connection_panel is not None else None
        width, height = dims if dims is not None else (1936, 1096)
        self._sensor_width, self._sensor_height = width, height
        self._demo_star_xy = (width * 0.4 + float(rng.uniform(-80, 80)), height * 0.5 + float(rng.uniform(-80, 80)))
        # Never exactly 0 -- a perfectly horizontal demo would defeat the
        # point of practicing on one that isn't.
        sign = 1.0 if rng.random() < 0.5 else -1.0
        self._demo_true_angle_deg = sign * float(rng.uniform(1.5, self._DEMO_ANGLE_RANGE_DEG))
        self._order0_xy = None
        self._trace_points = None
        self._measured_angle_deg = None
        self._mode = None
        self._view_xlim = None
        self._view_ylim = None
        self._render_frame()

    def _render_frame(self, frame_seed: int | None = None) -> None:
        if self._is_real():
            # Real frame from the connected camera (see LiveCameraFeed) --
            # None until the first preview_frame event arrives, in which
            # case there's nothing new to draw yet this tick.
            feed = self._live_camera_feed
            if feed.last_frame is None:
                return
            self._last_frame = feed.last_frame
            self._sensor_width, self._sensor_height = feed.width, feed.height
        else:
            self._last_frame = _synthetic_full_frame(
                seed=self._DEMO_SEED, sensor_width=self._sensor_width, sensor_height=self._sensor_height,
                star_xy=self._demo_star_xy, angle_deg=self._demo_true_angle_deg, brightness_scale=1.3,
                frame_seed=frame_seed,
            )
        self._redraw_current()

    def _live_tick(self) -> None:
        # Demo -> real edge checked EVERY tick, regardless of tab
        # visibility below -- a demo-measured angle/order0/trace must not
        # silently survive into real mode. get_trail_angle_deg() is read
        # unconditionally by both AcquisitionPanel and ReductionPanel, so
        # if this were gated behind winfo_ismapped() (like the redraw
        # below legitimately is), connecting the real camera while never
        # revisiting this tab would leave a stale demo angle baked into
        # REAL captured frames' straightening. The stale zoom window
        # (sized for the demo's sensor dims) is cleared too.
        is_real_now = self._is_real()
        if is_real_now and not self._live_mode_was_real:
            self._clear_marks()
            self._view_xlim = None
            self._view_ylim = None
        self._live_mode_was_real = is_real_now
        # Same self-paced idiom as AcquisitionPanel/FlatsPanel's own
        # _live_tick -- only redraws while the tab is actually visible.
        # In demo mode only the noise draw changes (star position/angle/
        # marks/zoom are untouched); in real mode this just picks up
        # whatever LiveCameraFeed currently holds (itself updated at
        # ~10Hz by App._pump_events, independent of this slower tick).
        if self.winfo_ismapped():
            self._live_frame_count += 1
            self._render_frame(frame_seed=self._DEMO_SEED * 100_003 + self._live_frame_count)
        self.after(self._LIVE_INTERVAL_MS, self._live_tick)

    def _on_toggle_order0_mode(self) -> None:
        self._mode = None if self._mode == "order0" else "order0"
        self._update_status()

    def _on_toggle_trace_mode(self) -> None:
        self._mode = None if self._mode == "trace" else "trace"
        self._update_status()

    def _on_canvas_press(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if self._mode == "order0":
            self._order0_xy = (event.xdata, event.ydata)
            self._mode = None
            self._redraw_current()
        elif self._mode == "trace":
            self._drag_start = (event.xdata, event.ydata)

    def _on_canvas_release(self, event) -> None:
        if self._mode != "trace" or self._drag_start is None:
            return
        if event.xdata is None or event.ydata is None:
            self._drag_start = None
            return
        p0, p1 = self._drag_start, (event.xdata, event.ydata)
        self._drag_start = None
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 5:
            return  # a click, not a drag -- too short to be a meaningful trace
        self._trace_points = (p0, p1)
        self._measured_angle_deg = angle_from_points(p0, p1)
        self._mode = None
        self._redraw_current()

    def _redraw_current(self) -> None:
        if self._last_frame is None:
            return
        height, width = self._last_frame.shape
        self._ax.clear()
        self._ax.imshow(self._last_frame, cmap="inferno", aspect="auto", extent=(0, width, height, 0))
        if self._order0_xy is not None:
            self._ax.plot(*self._order0_xy, marker="+", color=PALETTE.accent_ok, markersize=14, markeredgewidth=2)
        if self._trace_points is not None:
            (x0, y0), (x1, y1) = self._trace_points
            self._ax.plot([x0, x1], [y0, y1], color=PALETTE.accent_warn, linewidth=1.5)
        # Preserves whatever zoom _on_scroll last set, rather than
        # snapping back to the full frame on every mark/trace redraw --
        # only _on_new_demo_frame resets these to None (full extent).
        if self._view_xlim is None or self._view_ylim is None:
            self._view_xlim = (0.0, float(width))
            self._view_ylim = (float(height), 0.0)
        self._ax.set_xlim(*self._view_xlim)
        self._ax.set_ylim(*self._view_ylim)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_visible(False)
        self._canvas.draw()
        self._update_status()

    def _on_scroll(self, event) -> None:
        """Zoom the view in/out around the cursor -- view-only, doesn't
        touch _last_frame or any mark/trace coordinates, which are always
        in real sensor-pixel data space regardless of the current zoom
        (see imshow's own extent= above), so marking stays accurate at
        any zoom level. See _zoomed_view for the shared math."""
        if self._last_frame is None:
            return
        height, width = self._last_frame.shape
        result = _zoomed_view(event, self._ax.get_xlim(), self._ax.get_ylim(), width, height)
        if result is None:
            return
        self._view_xlim, self._view_ylim = result
        self._ax.set_xlim(*self._view_xlim)
        self._ax.set_ylim(*self._view_ylim)
        self._canvas.draw_idle()

    def _clear_marks(self) -> None:
        self._order0_xy = None
        self._trace_points = None
        self._measured_angle_deg = None
        self._mode = None

    def _on_reset(self) -> None:
        self._clear_marks()
        self._redraw_current()

    def _update_status(self) -> None:
        if self._is_real():
            self._mode_var.set("Mode: LIVE -- real camera connected (Connection tab).")
            self._new_demo_button.configure(state="disabled")
        else:
            self._mode_var.set("Mode: demo/practice -- a synthetic star, since no real camera is connected.")
            self._new_demo_button.configure(state="normal")
        # ✅/⬜ prefixes match the same convention AcquisitionPanel/
        # FlatsPanel use for their own session-status labels -- a glance
        # at the leading icon says "done" or "not yet" before reading any
        # of the text.
        self._order0_button.configure(text="Mark order 0 (click the frame)" if self._mode == "order0" else "Mark order 0")
        self._trace_button.configure(text="Draw trace (drag on the frame)" if self._mode == "trace" else "Draw trace")
        order0_note = (
            f"✅ Order 0 marked at ({self._order0_xy[0]:.0f}, {self._order0_xy[1]:.0f}). "
            if self._order0_xy is not None else "⬜ Order 0 not marked yet. "
        )
        if self._measured_angle_deg is None:
            trace_note = "⬜ 2. Click \"Draw trace\", then click-drag-release along the spectrum trail."
            self._status_var.set(order0_note + trace_note if self._order0_xy is not None else "⬜ 1. Click \"Mark order 0\", then click the star below.")
            self._status_label.configure(foreground=PALETTE.fg_dim)
            return
        angle = self._measured_angle_deg
        if abs(angle) > self._LARGE_ANGLE_WARNING_DEG:
            self._status_var.set(
                f"{order0_note}✅ Measured angle: {angle:+.1f}° -- fairly tilted. Consider re-rotating the Star "
                "Analyser in its locking ring for a more accurate spectrum (see its manual's own advice on this).",
            )
            self._status_label.configure(foreground=PALETTE.accent_warn)
        else:
            self._status_var.set(
                f"{order0_note}✅ Measured angle: {angle:+.1f}° -- feeds every Reference star/Target tab from here.",
            )
            self._status_label.configure(foreground=PALETTE.accent_ok)

    def get_trail_angle_deg(self) -> float | None:
        return self._measured_angle_deg


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
    -- see FlatsPanel.

    Framing is two stages, matching real practice: a FULL raw-sensor live
    view (real ASI290MC-sized, not a suspiciously pre-cropped little
    rectangle) to find the star and mark order 0 by hand -- reset
    whenever a new GOTO is issued, since the star lands somewhere new
    each time -- and, once marked, the small "already found" local
    preview this tab always showed, used for exposure/gain tuning, the
    uncalibrated profile, and capture. The trail's own tilt is measured
    ONCE in AlignmentPanel and shared here (and with the other
    AcquisitionPanel), not remeasured per star -- same "shared setup
    property, not per-star" split as flats vs. darks."""

    def __init__(
        self, parent: tk.Misc, role: str, seed: int, get_star, get_spectrum=None,
        mount_worker: MountWorker | None = None, connection_panel: ConnectionPanel | None = None,
        alignment_panel: AlignmentPanel | None = None, live_camera_feed: LiveCameraFeed | None = None,
        camera_worker: CameraWorker | None = None,
    ):
        super().__init__(parent, padding=10)
        self._role = role  # "reference" or "target"
        self._title = "Reference star" if role == "reference" else "Target"
        self._seed = seed
        self._get_star = get_star
        self._get_spectrum = get_spectrum
        self._mount_worker = mount_worker
        self._connection_panel = connection_panel
        self._alignment_panel = alignment_panel
        self._live_camera_feed = live_camera_feed
        self._camera_worker = camera_worker
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
        # them. Populated either by the synthetic generator (mock) or,
        # frame by frame as they arrive, by RealCaptureState.consume
        # (real camera, see self._real_capture) -- either way this is
        # the single list Reduction
        # reads from, so it doesn't need to know which one happened.
        self._science_frames: list[np.ndarray] = []
        self._dark_frames: list[np.ndarray] = []
        self._offset_frames: list[np.ndarray] = []
        self._capture_seed_counter = 0
        self._order0_full_xy: tuple[float, float] | None = None
        self._mark_mode = False
        self._last_full_frame: np.ndarray | None = None
        # None means "full sensor extent" -- see _redraw_full_frame, same
        # scroll-to-zoom idiom as AlignmentPanel (see _zoomed_view).
        self._full_frame_view_xlim: tuple[float, float] | None = None
        self._full_frame_view_ylim: tuple[float, float] | None = None
        # Real-camera capture is asynchronous -- see RealCaptureState's own
        # docstring for why (frames only arrive at ~10Hz, unlike the
        # instant synthetic generator).
        self._real_capture = RealCaptureState(live_camera_feed) if live_camera_feed is not None else None

        # Simulates imperfect GOTO pointing: the star doesn't land dead
        # center in the frame in real life -- where in the FULL sensor it
        # actually lands, see _star_full_xy. Grounded in a real pointing-
        # error magnitude (arcmin) converted to pixels via the real plate
        # scale (ConnectionPanel.get_plate_scale_arcsec_per_px, itself
        # from the real focal length) when available; falls back to an
        # arbitrary pixel range if focal length/pixel size aren't set to
        # something valid yet.
        pan_rng = np.random.default_rng(seed)
        plate_scale = self._connection_panel.get_plate_scale_arcsec_per_px() if self._connection_panel is not None else None
        if plate_scale is not None and plate_scale > 0:
            error_x_arcmin = float(pan_rng.uniform(-_TYPICAL_GOTO_ACCURACY_ARCMIN, _TYPICAL_GOTO_ACCURACY_ARCMIN))
            error_y_arcmin = float(pan_rng.uniform(-_TYPICAL_GOTO_ACCURACY_ARCMIN, _TYPICAL_GOTO_ACCURACY_ARCMIN))
            self._true_pan_x = error_x_arcmin * 60.0 / plate_scale
            self._true_pan_y = error_y_arcmin * 60.0 / plate_scale
        else:
            self._true_pan_x = float(pan_rng.uniform(-300.0, 300.0))
            self._true_pan_y = float(pan_rng.uniform(-300.0, 300.0))

        self._header_var = tk.StringVar(value=f"{self._title}: (none selected yet)")
        ttk.Label(self, textvariable=self._header_var, font=("", 10, "bold")).pack(anchor="w")

        columns = ttk.Frame(self)
        columns.pack(fill="both", expand=True, pady=(10, 0))
        left = ttk.Frame(columns)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(columns)
        right.pack(side="left", fill="none", padx=(10, 0))

        framing_frame = ttk.LabelFrame(
            left, text="Full-frame view -- find the star, then mark order 0 (scroll to zoom)", padding=8,
        )
        framing_frame.pack(fill="both", expand=True)
        self._full_frame_figure = Figure(figsize=(5.5, 2.4), dpi=100)
        self._full_frame_ax = self._full_frame_figure.add_subplot(111)
        self._full_frame_figure.patch.set_facecolor(PALETTE.bg)
        self._full_frame_canvas = FigureCanvasTkAgg(self._full_frame_figure, master=framing_frame)
        self._full_frame_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._full_frame_canvas.mpl_connect("button_press_event", self._on_full_frame_click)
        self._full_frame_canvas.mpl_connect("scroll_event", self._on_full_frame_scroll)
        framing_button_row = ttk.Frame(framing_frame)
        framing_button_row.pack(fill="x", pady=(6, 0))
        self._mark_button = ttk.Button(framing_button_row, text="Mark order 0", command=self._on_toggle_mark_mode)
        self._mark_button.pack(side="left")
        ttk.Button(framing_button_row, text="Refresh view", command=self._refresh_full_frame).pack(side="left", padx=(6, 0))
        self._order0_status_var = tk.StringVar(value="")
        ttk.Label(
            framing_frame, textvariable=self._order0_status_var, foreground=PALETTE.fg_dim, wraplength=480, justify="left",
        ).pack(anchor="w", pady=(4, 0))

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
        # See TabResyncTracker's own docstring -- pushes this panel's own
        # exposure/gain to the (shared) real camera on the right edges,
        # never while a capture is running anywhere.
        self._resync_tracker = TabResyncTracker()
        self._update_order0_status()
        self._on_settings_changed()
        self._refresh_header()
        self._refresh_full_frame()
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
        if self._real_capture is not None and self._real_capture.active:
            # A real capture crops each incoming frame around
            # self._order0_full_xy (see RealCaptureState.consume's
            # crop_fn) -- nulling it out from under an in-progress
            # capture would crash the next tick (extract_aligned_crop on
            # a None order0) and leave the capture stuck forever.
            self._goto_status_var.set("A capture is still in progress -- wait for it to finish before GOTO.")
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
        # The star lands somewhere new after every GOTO -- any previous
        # order-0 mark is stale, so make the operator re-find it rather
        # than silently keep extracting from the old (now wrong) spot.
        # The view is reset too, for the same reason a stale zoom window
        # centered on the OLD position wouldn't even show the new one.
        self._order0_full_xy = None
        self._full_frame_view_xlim = None
        self._full_frame_view_ylim = None
        self._update_order0_status()
        self._refresh_full_frame()

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

    def _is_real(self) -> bool:
        return self._live_camera_feed is not None and self._live_camera_feed.is_active

    def _exposure_ms(self) -> float:
        return 10.0 + self._exposure_var.get() * 20.0  # 10-2010ms -- display/model only, mock path

    def _gain_value(self) -> int:
        return round(self._gain_var.get() * 5.7)  # 0-570, matches this project's mock camera range

    def _on_settings_changed(self, _value: str | None = None) -> None:
        if self._is_real():
            exposure_us = self._live_camera_feed.slider_to_exposure_us(self._exposure_var.get())
            gain = self._live_camera_feed.slider_to_gain(self._gain_var.get())
            self._exposure_label_var.set(format_exposure_us(exposure_us))
            self._gain_label_var.set(str(gain))
            if self._camera_worker is not None:
                self._camera_worker.set_exposure_us(round(exposure_us))
                self._camera_worker.set_gain(gain)
        else:
            self._exposure_label_var.set(format_exposure_us(self._exposure_ms() * 1000.0))
            self._gain_label_var.set(str(self._gain_value()))
        self._dark_hint_var.set(f"Matches current settings: {self._exposure_label_var.get()}, gain {self._gain_label_var.get()}")
        self._offset_hint_var.set(f"Minimum exposure, gain {self._gain_label_var.get()} -- exposure length doesn't matter for bias")
        self._render_local_patch(frame_seed=None)

    def _live_tick(self) -> None:
        # Real-capture consumption runs regardless of tab visibility --
        # unlike the redraws below (purely cosmetic, fine to skip while
        # not looking at this tab), an in-progress capture must keep
        # collecting frames even if the operator switches to another tab
        # mid-capture, same as the synthetic capture path (which finishes
        # instantly and was never visibility-gated either).
        if self._real_capture is not None:
            self._real_capture.consume()
        # Redraws the LOCAL patch (not the full-frame view -- see this
        # class's own docstring for why that one is on-demand only IN
        # MOCK MODE, where synthesizing a full 1936x1096 frame costs
        # ~200ms, see _synthetic_full_frame's own docstring) with a fresh
        # noise/jitter draw every _LIVE_INTERVAL_MS, independent of any
        # control move -- see _synthetic_trail_image's docstring for why
        # this is what actually reads as "live" rather than a plot that
        # only ever changes when you touch a control. In REAL mode the
        # full-frame view is cheap to refresh too (just redrawing whatever
        # LiveCameraFeed already decoded, no generation cost), so it
        # live-updates here as well, unlike the mock's on-demand-only
        # full-frame view.
        is_real_now = self._is_real()
        active_count = self._live_camera_feed.active_capture_count if self._live_camera_feed is not None else 0
        if self._resync_tracker.update(self.winfo_ismapped(), is_real_now, active_count):
            self._on_settings_changed()
        if self.winfo_ismapped():
            self._live_frame_count += 1
            if is_real_now:
                self._refresh_full_frame()
            self._render_local_patch(frame_seed=self._seed * 100_003 + self._live_frame_count)
            self._update_order0_status()
            self._stats_var.set(f"fps: {1000.0 / _LIVE_INTERVAL_MS:.1f}   frames: {self._live_frame_count}")
        self.after(_LIVE_INTERVAL_MS, self._live_tick)

    def _current_capture_params(self) -> tuple[float, tuple[np.ndarray, np.ndarray] | None]:
        """(brightness, spectrum) for whatever the exposure/gain controls
        are set to RIGHT NOW -- the single place the live preview, the
        full-frame view, and the actual frame capture all read these
        from, so a captured frame always matches what the operator was
        just looking at when they clicked the button."""
        brightness = 0.4 + 1.2 * (self._exposure_var.get() / 100.0) * (0.5 + self._gain_var.get() / 100.0)
        spectrum = self._get_spectrum() if self._get_spectrum is not None else None
        return brightness, spectrum

    def _next_capture_seed(self) -> int:
        self._capture_seed_counter += 1
        return self._seed * 1_000_003 + self._capture_seed_counter

    def _current_dispersion_a(self) -> float | None:
        return self._connection_panel.get_dispersion_a_per_px() if self._connection_panel is not None else None

    def _current_angle_deg(self) -> float:
        """The shared trail tilt from AlignmentPanel -- 0.0 (horizontal)
        if that hasn't been measured yet, same "not wrong, just the
        least-assuming fallback" spirit as spectro.reduction.
        assumed_dispersion's own fallback."""
        if self._alignment_panel is None:
            return 0.0
        angle = self._alignment_panel.get_trail_angle_deg()
        return angle if angle is not None else 0.0

    def _star_full_xy(self) -> tuple[float, float]:
        """Where this star actually lands in the full sensor -- the
        nominal GOTO aim point (frame center) plus this panel's own
        simulated pointing error (_true_pan_x/y)."""
        dimensions = self._connection_panel.get_sensor_dimensions() if self._connection_panel is not None else None
        width, height = dimensions if dimensions is not None else (1936, 1096)
        return width / 2.0 + self._true_pan_x, height / 2.0 + self._true_pan_y

    def _refresh_full_frame(self) -> None:
        if self._is_real():
            # The real camera's own preview stream (see LiveCameraFeed) --
            # nothing to do yet if no frame has arrived since connecting;
            # the next tick of whatever called this (e.g. _live_tick, or
            # App's own periodic pump) will pick it up once one does.
            last_frame = self._live_camera_feed.last_frame
            if last_frame is None:
                return
            self._last_full_frame = last_frame
            self._redraw_full_frame()
            return
        dimensions = self._connection_panel.get_sensor_dimensions() if self._connection_panel is not None else None
        width, height = dimensions if dimensions is not None else (1936, 1096)
        brightness, spectrum = self._current_capture_params()
        self._last_full_frame = _synthetic_full_frame(
            seed=self._seed, sensor_width=width, sensor_height=height, star_xy=self._star_full_xy(),
            angle_deg=self._current_angle_deg(), spectrum=spectrum, brightness_scale=brightness,
            frame_seed=self._next_capture_seed(), dispersion_a=self._current_dispersion_a(),
        )
        self._redraw_full_frame()

    def _redraw_full_frame(self) -> None:
        if self._last_full_frame is None:
            return
        height, width = self._last_full_frame.shape
        self._full_frame_ax.clear()
        self._full_frame_ax.imshow(self._last_full_frame, cmap="inferno", aspect="auto", extent=(0, width, height, 0))
        if self._order0_full_xy is not None:
            self._full_frame_ax.plot(
                *self._order0_full_xy, marker="+", color=PALETTE.accent_ok, markersize=14, markeredgewidth=2,
            )
        # Preserves whatever zoom _on_full_frame_scroll last set, same
        # idiom as AlignmentPanel._redraw_current -- only _on_goto resets
        # these to None (full extent), since that's the only time the
        # star actually moves to somewhere a stale zoom wouldn't show.
        if self._full_frame_view_xlim is None or self._full_frame_view_ylim is None:
            self._full_frame_view_xlim = (0.0, float(width))
            self._full_frame_view_ylim = (float(height), 0.0)
        self._full_frame_ax.set_xlim(*self._full_frame_view_xlim)
        self._full_frame_ax.set_ylim(*self._full_frame_view_ylim)
        self._full_frame_ax.set_xticks([])
        self._full_frame_ax.set_yticks([])
        for spine in self._full_frame_ax.spines.values():
            spine.set_visible(False)
        self._full_frame_canvas.draw()

    def _on_full_frame_scroll(self, event: object) -> None:
        """Zoom the full-frame view in/out around the cursor -- see
        AlignmentPanel._on_scroll and _zoomed_view for the shared math;
        same reasoning applies here (view-only, marking stays accurate at
        any zoom level since clicks are read in real data coordinates)."""
        if self._last_full_frame is None:
            return
        height, width = self._last_full_frame.shape
        result = _zoomed_view(event, self._full_frame_ax.get_xlim(), self._full_frame_ax.get_ylim(), width, height)
        if result is None:
            return
        self._full_frame_view_xlim, self._full_frame_view_ylim = result
        self._full_frame_ax.set_xlim(*self._full_frame_view_xlim)
        self._full_frame_ax.set_ylim(*self._full_frame_view_ylim)
        self._full_frame_canvas.draw_idle()

    def _on_toggle_mark_mode(self) -> None:
        if not self._mark_mode and self._real_capture is not None and self._real_capture.active:
            # Re-marking mid-capture would silently mix crop anchors
            # within one batch -- frames collected before and after the
            # re-mark would be stacked together as if pixel-aligned, with
            # no error (see RealCaptureState.consume's crop_fn, which
            # re-reads self._order0_full_xy fresh on every tick).
            self._order0_status_var.set("⚠ A capture is still in progress -- wait for it to finish before re-marking order 0.")
            return
        self._mark_mode = not self._mark_mode
        self._mark_button.configure(text="Click the star above..." if self._mark_mode else "Mark order 0")

    def _on_full_frame_click(self, event: object) -> None:
        if not self._mark_mode or event.xdata is None or event.ydata is None:  # type: ignore[attr-defined]
            return
        if self._real_capture is not None and self._real_capture.active:
            return  # same guard as _on_toggle_mark_mode -- shouldn't be reachable, but don't mark if it is
        self._order0_full_xy = (event.xdata, event.ydata)  # type: ignore[attr-defined]
        self._mark_mode = False
        self._mark_button.configure(text="Mark order 0")
        self._redraw_full_frame()
        self._update_order0_status()
        self._on_settings_changed()  # the local patch is only meaningful from here on -- refresh it now

    def _update_order0_status(self) -> None:
        mode_note = "LIVE, real camera. " if self._is_real() else "Mock/synthetic. "
        if self._order0_full_xy is None:
            self._order0_status_var.set(
                mode_note + "Not marked yet -- click \"Mark order 0\", then click the star in the frame above.",
            )
        else:
            x, y = self._order0_full_xy
            self._order0_status_var.set(f"{mode_note}Order 0 marked at ({x:.0f}, {y:.0f}) px.")

    def _render_local_patch(self, frame_seed: int | None) -> None:
        if self._order0_full_xy is None:
            self._clear_local_patch()
            return
        dispersion_a = self._current_dispersion_a()
        if self._is_real():
            full_frame = self._live_camera_feed.last_frame
            if full_frame is None:
                self._clear_local_patch()
                return
            # Crops around order0 AND straightens by the real measured
            # angle (extract_aligned_crop derotates BY angle_deg, unlike
            # the synthetic generator below where angle_deg=0.0 means
            # "paint it already straight") -- same reasoning as the mock
            # branch's own comment: the extraction band/profile below
            # assume a horizontal trail, so this previews what the
            # operator gets after ReductionPanel's own correction step,
            # not the real (still tilted) raw view. The captured frames
            # themselves (_on_capture_science etc., via RealCaptureState)
            # crop WITHOUT derotating -- tilt preserved, corrected once
            # during Reduction, not per frame -- see those methods' own
            # comments.
            image = extract_aligned_crop(full_frame, self._order0_full_xy, self._current_angle_deg())
        else:
            brightness, spectrum = self._current_capture_params()
            # angle_deg=0.0, NOT self._current_angle_deg() -- this preview
            # exists to judge exposure/gain from the extraction band + profile
            # below (_draw_extraction_band/_draw_profile), both of which
            # assume a horizontal trail (same TRAIL_ROW spectro/reduction.py
            # uses). Showing it already straightened previews what the
            # operator will actually get after ReductionPanel's own
            # extract_aligned_crop step, rather than a tilted trail crossing a
            # horizontal band that would misjudge SNR. The captured frames
            # themselves (_on_capture_science etc.) still use the real
            # measured angle -- only this display is straightened.
            image = _synthetic_trail_image(
                seed=self._seed, brightness_scale=brightness, spectrum=spectrum, frame_seed=frame_seed,
                dispersion_a=dispersion_a, angle_deg=0.0,
            )
        self._preview_ax.clear()
        self._preview_ax.imshow(image, cmap="inferno", aspect="auto")
        _draw_extraction_band(self._preview_ax, image.shape[1])
        self._preview_ax.set_xlim(-0.5, image.shape[1] - 0.5)
        self._preview_ax.set_ylim(image.shape[0] - 0.5, -0.5)
        self._preview_ax.set_xticks([])
        self._preview_ax.set_yticks([])
        for spine in self._preview_ax.spines.values():
            spine.set_visible(False)
        self._preview_canvas.draw()
        _draw_profile(self._profile_ax, self._profile_figure, image, dispersion_a)
        self._profile_canvas.draw()
        _draw_histogram(self._hist_ax, self._hist_figure, image, compact=True)
        self._hist_canvas.draw()

    def _clear_local_patch(self) -> None:
        for ax, canvas in ((self._preview_ax, self._preview_canvas), (self._profile_ax, self._profile_canvas)):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            canvas.draw()
        self._preview_ax.text(
            0.5, 0.5, "Mark order 0 in the full-frame view above first", ha="center", va="center",
            color=PALETTE.fg_dim, fontsize=8, transform=self._preview_ax.transAxes, wrap=True,
        )
        self._preview_canvas.draw()
        self._hist_ax.clear()
        for spine in self._hist_ax.spines.values():
            spine.set_visible(False)
        self._hist_canvas.draw()

    def _on_capture_science(self) -> None:
        if self._order0_full_xy is None:
            self._science_var.set("⚠ Mark order 0 in the full-frame view above first.")
            return
        # Multiple REAL frames per click, not one -- each with its own
        # independent noise draw (a fresh frame_seed) so stacking them
        # later (see spectro/reduction.py's stack_frames) actually
        # improves SNR over a single frame, same reason darks/offset are
        # already batched.
        n = max(1, self._science_frames_var.get())
        if self._is_real():
            def finalize() -> None:
                self._science_count += n
                self._science_var.set(f"✅ {self._title} spectrum ({self._science_count} frames)")
            # Cropped around order0 but NOT derotated (angle_deg=0.0,
            # unlike the live-preview branch in _render_local_patch) --
            # the real tilt is preserved here, corrected once during
            # Reduction, not per captured frame, same "correct once, not
            # per-frame" design the synthetic path below already follows.
            started = self._real_capture.start(
                self._science_frames, n, self._science_var, f"{self._title.lower()} spectrum",
                lambda frame: extract_aligned_crop(frame, self._order0_full_xy, 0.0), finalize,
            )
            if not started:
                self._science_var.set("⚠ A capture is already in progress -- wait for it to finish.")
            return
        brightness, spectrum = self._current_capture_params()
        dispersion_a = self._current_dispersion_a()
        angle_deg = self._current_angle_deg()
        for _ in range(n):
            frame = _synthetic_trail_image(
                seed=self._seed, brightness_scale=brightness, spectrum=spectrum,
                frame_seed=self._next_capture_seed(), dispersion_a=dispersion_a, angle_deg=angle_deg,
            )
            self._science_frames.append(frame)
        self._science_count += n
        self._science_var.set(f"✅ {self._title} spectrum ({self._science_count} frames)")

    def _on_capture_dark(self) -> None:
        if self._is_real():
            # Real darks still need cropping around order0 (so they align
            # pixel-for-pixel with science for the science-dark subtraction
            # in spectro/reduction.py's calibrate_science) -- unlike the
            # synthetic path below, where an include_signal=False patch is
            # position-independent uniform noise.
            if self._order0_full_xy is None:
                self._dark_var.set("⚠ Mark order 0 in the full-frame view above first.")
                return
            def finalize() -> None:
                self._dark_count += 20
                self._dark_var.set(f"✅ Darks ({self._dark_count} frames)")
            started = self._real_capture.start(
                self._dark_frames, 20, self._dark_var, "darks",
                lambda frame: extract_aligned_crop(frame, self._order0_full_xy, 0.0), finalize,
            )
            if not started:
                self._dark_var.set("⚠ A capture is already in progress -- wait for it to finish.")
            return
        # No star signal at all (cap on) -- brightness_scale/angle are
        # irrelevant here since include_signal=False skips the only
        # things they'd have affected.
        for _ in range(20):
            frame = _synthetic_trail_image(seed=self._seed, frame_seed=self._next_capture_seed(), include_signal=False)
            self._dark_frames.append(frame)
        self._dark_count += 20
        self._dark_var.set(f"✅ Darks ({self._dark_count} frames)")

    def _on_capture_offset(self) -> None:
        if self._is_real():
            if self._order0_full_xy is None:
                self._offset_var.set("⚠ Mark order 0 in the full-frame view above first.")
                return
            # previous_exposure_us is just a read (slider_to_exposure_us),
            # not a hardware write -- safe to compute before knowing
            # whether start() will actually arm the capture.
            previous_exposure_us = self._live_camera_feed.slider_to_exposure_us(self._exposure_var.get())
            def restore_exposure() -> None:
                if self._camera_worker is not None:
                    self._camera_worker.set_exposure_us(round(previous_exposure_us))
            def finalize() -> None:
                self._offset_count += 20
                self._offset_var.set(f"✅ Offset/bias ({self._offset_count} frames)")
                restore_exposure()
            # Same started-bool pattern as science/dark/flats (RealCaptureState.
            # start() already refuses if a capture is active) -- letting it be
            # the single guard means minimum exposure is only ever forced
            # below once a capture actually armed, not wasted on a refusal.
            started = self._real_capture.start(
                self._offset_frames, 20, self._offset_var, "offset/bias",
                lambda frame: extract_aligned_crop(frame, self._order0_full_xy, 0.0), finalize,
                on_abort=restore_exposure,
            )
            if not started:
                self._offset_var.set("⚠ A capture is already in progress -- wait for it to finish.")
                return
            # Minimum exposure for a real bias/offset frame -- matches the
            # hint text below ("exposure length doesn't matter for bias").
            # Gain is left as-is (bias level depends mostly on gain, not
            # exposure -- same assumption calibrate_science's own
            # docstring already documents for the flat-bias approximation).
            # The PREVIOUS (slider-implied) exposure is restored either
            # way (finalize on success, on_abort if the camera disconnects
            # mid-capture) -- without this, a science/dark capture right
            # after would silently run at minimum exposure instead of what
            # the UI still shows.
            min_exposure_us, _ = self._live_camera_feed.get_control_range("Exposure", 32.0, 2_000_000_000.0)
            if self._camera_worker is not None:
                self._camera_worker.set_exposure_us(round(min_exposure_us))
            return
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

    def get_order0_full_xy(self) -> tuple[float, float] | None:
        """Where THIS star's order0 was marked in full-sensor coordinates
        -- used by ReductionPanel to crop a full-frame master flat (real
        mode) to the same sensor region this panel's own science/dark/
        offset frames were cropped from, see _flat_for_calibration."""
        return self._order0_full_xy


# -- Flats tab ---------------------------------------------------------------


class FlatsPanel(ttk.Frame):
    """Flat frames correct the optical train + sensor (vignetting, pixel-
    to-pixel sensitivity, dust) -- NOT tied to a specific star, so unlike
    darks/offset (see AcquisitionPanel) these only need doing once per
    setup, not once per reference/target. A good flat needs the exposure
    tuned so the histogram peak sits at ~2/3 of full well -- doing that
    from a plain "Capture 20" button with no live feedback (the previous
    single Calibration tab) isn't actually usable in practice, hence the
    live preview + histogram here.

    Real-mode capture stores the FULL, uncropped sensor frame -- flats
    have no star/order0 of their own to align to (taken against blank
    twilight sky/panel), and reference/target stars can land at
    different order0 positions on different GOTOs, so a single flat
    fixed-cropped at capture time could never be pixel-aligned with both.
    Instead each science/dark frame's own order0 crops the SAME master
    flat again, freshly, at calibration time -- see ReductionPanel.
    _flat_for_calibration -- so the flat that gets divided into a given
    star's science frame always comes from the physically-correct sensor
    region for THAT star, not a generic frame-center guess."""

    def __init__(self, parent: tk.Misc, live_camera_feed: LiveCameraFeed | None = None, camera_worker: CameraWorker | None = None):
        super().__init__(parent, padding=10)
        self._flat_count = 0
        self._flat_frames: list[np.ndarray] = []
        self._master_flat: np.ndarray | None = None
        self._capture_seed_counter = 0
        self._live_camera_feed = live_camera_feed
        self._camera_worker = camera_worker
        self._real_capture = RealCaptureState(live_camera_feed) if live_camera_feed is not None else None

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
        # See TabResyncTracker's own docstring, and AcquisitionPanel's
        # copy of this -- resyncs this tab's own exposure to the shared
        # real camera on the right edges, never mid-capture.
        self._resync_tracker = TabResyncTracker()
        self._on_settings_changed()
        self._live_tick()

    def _is_real(self) -> bool:
        return self._live_camera_feed is not None and self._live_camera_feed.is_active

    def _crop_from_center(self, full_frame: np.ndarray) -> np.ndarray:
        height, width = full_frame.shape
        return extract_aligned_crop(full_frame, (width / 2.0, height / 2.0), 0.0)

    def _on_settings_changed(self, _value: str | None = None) -> None:
        if self._is_real() and self._camera_worker is not None:
            self._camera_worker.set_exposure_us(round(self._live_camera_feed.slider_to_exposure_us(self._exposure_var.get())))
        self._render_frame(frame_seed=None)

    def _live_tick(self) -> None:
        # Real-capture consumption runs regardless of tab visibility, same
        # reasoning as AcquisitionPanel._live_tick's own copy of this --
        # an in-progress capture must keep collecting frames even if the
        # operator switches to another tab mid-capture.
        if self._real_capture is not None:
            self._real_capture.consume()
        # Same self-paced idiom as AcquisitionPanel._live_tick -- see its
        # docstring for why a periodic redraw (not just on slider moves)
        # is what actually reads as a live feed.
        is_real_now = self._is_real()
        active_count = self._live_camera_feed.active_capture_count if self._live_camera_feed is not None else 0
        if self._resync_tracker.update(self.winfo_ismapped(), is_real_now, active_count):
            self._on_settings_changed()
        if self.winfo_ismapped():
            self._live_frame_count += 1
            self._render_frame(frame_seed=3 * 100_003 + self._live_frame_count)
        self.after(_LIVE_INTERVAL_MS, self._live_tick)

    def _render_frame(self, frame_seed: int | None) -> None:
        if self._is_real():
            full_frame = self._live_camera_feed.last_frame
            if full_frame is None:
                return
            image = self._crop_from_center(full_frame)
        else:
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

    def _expected_flat_shape(self) -> tuple[int, int]:
        if self._is_real():
            return self._live_camera_feed.height, self._live_camera_feed.width
        return 90, 420

    def _on_capture(self) -> None:
        expected_shape = self._expected_flat_shape()
        if self._flat_frames and self._flat_frames[0].shape != expected_shape:
            # Switched mock<->real (or reconnected to a differently-sized
            # real camera) since the last capture -- those old frames are
            # a different shape/physical sensor region and can't be
            # stacked with new ones (np.stack in stack_frames requires
            # uniform shapes), so start this batch fresh rather than let
            # "Build master flat" crash with an unhandled ValueError.
            self._flat_frames = []
            self._flat_count = 0
            self._flat_var.set("⬜ Flats (0 frames) -- previous flats discarded (camera mode/size changed)")
        if self._is_real():
            def finalize() -> None:
                self._flat_count += 20
                self._flat_var.set(f"✅ Flats ({self._flat_count} frames)")
            # Stores the FULL frame, uncropped (crop_fn is identity) --
            # NOT _crop_from_center. A flat corrects per-pixel sensor
            # sensitivity, so it must be cropped to match wherever a
            # given science/dark frame's own order0 landed at calibration
            # time (see ReductionPanel._flat_for_calibration), not to a
            # fixed frame-center window that would silently misalign with
            # order0 the moment it isn't literally at the sensor's
            # center -- the normal case once a real GOTO is involved.
            started = self._real_capture.start(self._flat_frames, 20, self._flat_var, "flats", lambda frame: frame, finalize)
            if not started:
                self._flat_var.set("⚠ A capture is already in progress -- wait for it to finish.")
            return
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
        flats_panel: FlatsPanel, target_panel: TargetPanel, connection_panel: ConnectionPanel,
        alignment_panel: AlignmentPanel, session: Session,
    ):
        super().__init__(parent, padding=10)
        self._reference_panel = reference_panel
        self._target_capture_panel = target_capture_panel
        self._flats_panel = flats_panel
        self._target_panel = target_panel
        self._connection_panel = connection_panel
        self._alignment_panel = alignment_panel
        self._session = session

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

    def _flat_for_calibration(
        self, master_flat: np.ndarray, science_shape: tuple[int, int], order0_xy: tuple[float, float] | None,
    ) -> np.ndarray:
        """The flat to actually divide into THIS star's science/dark
        stack -- as-is if it's already the same local-patch shape (mock
        mode, or an already-matching real flat), or freshly cropped
        around this star's own order0 if it's a full sensor frame (real
        mode -- see FlatsPanel's own docstring for why flats are now
        stored uncropped: a single fixed crop couldn't be pixel-aligned
        with both the reference and target panels' own, potentially
        different, order0 positions at once)."""
        if master_flat.shape == science_shape:
            return master_flat
        if order0_xy is None:
            raise ReductionError(
                "the master flat is a full sensor frame but this star has no order0 marked -- "
                "can't align them (mark order 0 in its Reference star/Target tab first)",
            )
        return extract_aligned_crop(master_flat, order0_xy, 0.0, crop_shape=science_shape)

    def _on_build_masters(self) -> None:
        # Both stars are normally already chosen by the time masters are
        # built (captures happen first) -- this is where the session
        # folder gets created and named, see Session.ensure's own
        # docstring for why here rather than at the very first capture.
        reference_star = self._target_panel.get_reference_star()
        target_star = self._target_panel.get_target_star()
        session_dir = self._session.ensure(
            reference_star.name if reference_star is not None else None,
            target_star.name if target_star is not None else None,
        )

        def _star_info(star) -> dict:
            if star is None:
                return {"name": None}
            return {"name": star.name, "ra_deg": star.ra_deg, "dec_deg": star.dec_deg, "vmag": star.vmag}

        self._session.write_metadata({
            "reference_star": _star_info(reference_star),
            "target_star": _star_info(target_star),
            "trail_angle_deg": self._alignment_panel.get_trail_angle_deg(),
            "instrument": self._connection_panel.get_instrument_metadata(),
        })

        lines = [f"Session folder: {session_dir}"]
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
            self._session.save_fits(f"{label.lower()}_master_dark.fits", master_dark)
            self._session.save_fits(f"{label.lower()}_master_offset.fits", master_offset)
            self._session.save_fits_cube(f"{label.lower()}_dark_raw.fits", darks)
            self._session.save_fits_cube(f"{label.lower()}_offset_raw.fits", offsets)
            lines.append(
                f"{label}: {len(darks)} darks + {len(offsets)} offset frames stacked "
                f"(bias level ~{master_offset.mean():.1f} ADU).",
            )
        master_flat = self._flats_panel.get_master_flat()
        if master_flat is None:
            lines.append("Flats: build a master flat in the Flats tab first.")
            ok = False
        else:
            self._session.save_fits("master_flat.fits", master_flat)
            self._session.save_fits_cube("flats_raw.fits", self._flats_panel.get_flat_frames())
            lines.append(f"Flats: master flat ready (mean {master_flat.mean():.1f} ADU).")
        self._set_status(self._step1_status_label, self._step1_status_var, "\n".join(lines), ok)

    def _plot_profile_with_lines(
        self, ax, figure, canvas, profile: np.ndarray, detected: list, dispersion_a: float | None = None,
    ) -> None:
        ax.clear()
        pixels = np.arange(len(profile))
        ax.plot(
            pixels[_TRAIL_START_PX:_TRAIL_END_PX], profile[_TRAIL_START_PX:_TRAIL_END_PX],
            color=PALETTE.accent, linewidth=1,
        )
        detected_px = {label: px for label, _wl, px in detected}
        for label, wl in REFERENCE_LINES:
            assumed_px = _assumed_px_for_wavelength(wl, dispersion_a)
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
            flat = self._flat_for_calibration(master_flat, stack.shape, self._reference_panel.get_order0_full_xy())
            calibrated = calibrate_science(stack, self._reference_master_dark, flat, self._reference_master_offset)
        except ReductionError as exc:
            self._set_status(self._step2_status_label, self._step2_status_var, f"Reduction failed: {exc}", False)
            return
        # Straighten the trail's real tilt (see AlignmentPanel) BEFORE any
        # of the pipeline below, which has always assumed a perfectly
        # horizontal trail -- 0.0 (untouched) if alignment hasn't been
        # measured yet, exactly matches identity so this is always safe
        # to apply.
        angle_deg = self._alignment_panel.get_trail_angle_deg() or 0.0
        calibrated = extract_aligned_crop(
            calibrated, (_ROI_TARGET_X, _ROI_TARGET_Y), angle_deg,
            crop_shape=calibrated.shape, local_anchor=(_ROI_TARGET_X, _ROI_TARGET_Y),
        )
        self._reference_calibrated = calibrated
        self._session.save_fits("reference_calibrated.fits", calibrated)
        self._session.save_fits_cube("reference_science_raw.fits", science)
        profile = extract_profile(calibrated)
        assumed_dispersion_a = self._connection_panel.get_dispersion_a_per_px()
        detected = detect_line_pixels(profile, dispersion_a=assumed_dispersion_a)
        self._dispersion = fit_dispersion(detected)
        gain = snr_gain(len(science))
        angle_note = f" (trail straightened by {angle_deg:+.1f}°)" if abs(angle_deg) > 0.05 else ""
        if self._dispersion is not None:
            a, _b = self._dispersion
            msg = (
                f"{len(science)} frames stacked (SNR x{gain:.1f}){angle_note}. Detected {len(detected)}/{len(REFERENCE_LINES)} "
                f"reference lines -- fitted dispersion {a:.3f} Å/px."
            )
        elif assumed_dispersion_a is not None:
            msg = (
                f"{len(science)} frames stacked (SNR x{gain:.1f}){angle_note}. Only {len(detected)} line(s) detected -- "
                f"need at least 2 to fit a real dispersion, falling back to the Grating/distance-predicted "
                f"{assumed_dispersion_a:.2f} Å/px."
            )
        else:
            msg = (
                f"{len(science)} frames stacked (SNR x{gain:.1f}){angle_note}. Only {len(detected)} line(s) detected -- "
                "need at least 2 to fit a real dispersion, falling back to the assumed one."
            )
        self._set_status(self._step2_status_label, self._step2_status_var, msg, True)
        self._plot_profile_with_lines(
            self._step2_ax, self._step2_figure, self._step2_canvas, profile, detected, assumed_dispersion_a,
        )

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
            flat = self._flat_for_calibration(master_flat, stack.shape, self._target_capture_panel.get_order0_full_xy())
            calibrated = calibrate_science(stack, self._target_master_dark, flat, self._target_master_offset)
        except ReductionError as exc:
            self._set_status(self._step3_status_label, self._step3_status_var, f"Reduction failed: {exc}", False)
            return
        angle_deg = self._alignment_panel.get_trail_angle_deg() or 0.0
        calibrated = extract_aligned_crop(
            calibrated, (_ROI_TARGET_X, _ROI_TARGET_Y), angle_deg,
            crop_shape=calibrated.shape, local_anchor=(_ROI_TARGET_X, _ROI_TARGET_Y),
        )
        self._target_calibrated = calibrated
        self._session.save_fits("target_calibrated.fits", calibrated)
        self._session.save_fits_cube("target_science_raw.fits", science)
        profile = extract_profile(calibrated)
        gain = snr_gain(len(science))
        angle_note = f" (trail straightened by {angle_deg:+.1f}°)" if abs(angle_deg) > 0.05 else ""
        msg = (
            f"{len(science)} frames stacked (SNR x{gain:.1f}){angle_note}. Target's own lines are unknown -- "
            "nothing to detect yet."
        )
        self._set_status(self._step3_status_label, self._step3_status_var, msg, True)
        self._plot_profile_with_lines(
            self._step3_ax, self._step3_figure, self._step3_canvas, profile, [],
            self._connection_panel.get_dispersion_a_per_px(),
        )

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
        assumed_dispersion_a = self._connection_panel.get_dispersion_a_per_px()
        _pixels, _wavelengths, response, _measured = compute_response(
            self._reference_calibrated, ref_wl, ref_flux, self._dispersion, assumed_dispersion_a,
        )
        final_wl, final_flux = apply_response(self._target_calibrated, response, self._dispersion, assumed_dispersion_a)
        self._final_wl, self._final_flux = final_wl, final_flux
        self._session.save_spectrum_fits("final_spectrum.fits", final_wl, final_flux)
        self._result_version += 1
        if self._dispersion is not None:
            dispersion_note = f"real fitted dispersion ({self._dispersion[0]:.3f} Å/px)"
        elif assumed_dispersion_a is not None:
            dispersion_note = (
                f"Grating/distance-predicted dispersion ({assumed_dispersion_a:.2f} Å/px, not enough "
                "lines detected in step 2 to refit)"
            )
        else:
            dispersion_note = "assumed linear dispersion (not enough lines detected in step 2 to refit)"
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
                "Refused: Mock is selected, or mount/camera aren't both actually connected as real "
                "hardware, in the Connection tab. Connect both for real before exporting for AVSpec "
                "submission -- see ConnectionPanel.is_mock's own docstring for exactly what this checks.",
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
