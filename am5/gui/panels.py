"""ttk.Frame panels making up the GUI: connection, pass browser, manual
control, and transit (mount tracking + camera capture combined, the screen
used during an actual pass). Every one of them talks to the mount or camera
exclusively through a shared worker (see worker.py / camera/worker.py) —
never touches a Mount or camera device directly, and never calls a Tkinter
method from a background thread.
"""

from __future__ import annotations

import dataclasses
import math
import queue
import random
import threading
import time
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import numpy as np
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skyfield.api import wgs84
from skyfield.toposlib import GeographicPosition
from tkintermapview import TkinterMapView

from am5.angles import circular_diff_hours, equatorial_series_to_altaz, equatorial_to_altaz
from am5.clock_sync import ClockSyncStatus, check_clock_sync
from am5.constellations import constellations_altaz
from am5.ephemeris import PassWindow, Trajectory, compute_trajectory, find_passes, load_satellite_tle, meridian_crossings
from am5.optics import (
    DEFAULT_FULL_WELL_ELECTRONS,
    OpticalTrain,
    estimate_iss_magnitude,
    estimate_signal_electrons,
    max_exposure_s,
    render_iss_photo,
    suggest_gain,
)
from am5.gui.theme import PALETTE, style_axes
from am5.gui.worker import MountWorker, WorkerEvent
from am5.tracker import AxisSigns, LiveOffsets, TrackingConfig, decompose_error
from camera.finder import MAX_FINDER_EXPOSURE_US, FinderState, downsample_for_display
from camera.guiding import BlobDetection, GuidingCalibration, calibrate_from_nudges, detect_brightest_blob
from camera.ser_reader import SerReader
from camera.worker import CameraEvent, CameraWorker, frame_to_pgm, pgm_to_array

TLE_CACHE_DIR = Path("logs")

# PassesPanel's target picker: name -> (NORAD catalog number, magnitude_ref).
# magnitude_ref feeds am5.optics.estimate_iss_magnitude's distance scaling,
# which is specifically calibrated from a real ISS capture (see
# am5/optics.py) -- meaningless for any other object's actual size/
# reflectivity, so every non-ISS entry uses None (shown as "N/A" in the
# passes table, see PassesPanel._populate_tree) rather than a fabricated
# number. NORAD IDs verified against Celestrak, not typed from memory.
KNOWN_SATELLITES: dict[str, tuple[int, float | None]] = {
    "ISS (ZARYA)": (25544, -1.8),
    "Tiangong (CSS)": (48274, None),
}
CUSTOM_SATELLITE_LABEL = "Custom NORAD ID..."


def _local_and_utc(dt: datetime) -> str:
    """'HH:MM:SS local (HH:MM:SS UTC)' -- pass times are computed and stored
    in UTC (skyfield's native convention), but the operator is watching the
    real sky on their own local clock. Showing only UTC (or an unlabeled
    bare HH:MM:SS, which reads as local at a glance) is exactly how a
    session gets started hours off from the actual pass -- this is the
    fix for that after it happened for real."""
    return f"{dt.astimezone().strftime('%H:%M:%S')} local ({dt.strftime('%H:%M:%S')} UTC)"


def _meridian_detail_line(crossings: list, window: PassWindow) -> str:
    """Not just "will it flip" but "when" -- how far into the tracking
    session it happens (the actionable number: how long you have before you
    need to be watching the mount) and how far from culmination (context:
    crossings right at culmination are the ones worth double-checking tube
    clearance for, since that's also peak elevation)."""
    if not crossings:
        return "No meridian crossing during this pass"
    crossing_t = crossings[0]
    since_rise_s = (crossing_t - window.t_rise).total_seconds()
    from_culm_s = (crossing_t - window.t_culminate).total_seconds()
    culm_word = "after" if from_culm_s >= 0 else "before"
    return (f"MERIDIAN CROSSING at {crossing_t.strftime('%H:%M:%S')} UTC -- "
            f"{since_rise_s:.0f}s after tracking starts, {abs(from_culm_s):.0f}s {culm_word} culmination -- "
            f"pick a starting pier side that avoids a flip mid-pass")

MAX_PREVIEW_ZOOM = 4  # cap how many display pixels one sensor pixel becomes when
# magnifying a small ROI to fill the canvas -- avoids giant blocky pixels
MAX_EXPOSURE_SLIDER_US = 1_000_000  # 1s -- the slider covers our actual use
# case (sub-2ms ISS exposures) usefully; a real camera's nominal max can be
# far higher (mock reports up to 2000s) but that's not worth compressing the
# useful range for on a log slider, so we just clamp what the slider reaches.


def format_exposure_us(microseconds: float) -> str:
    if microseconds >= 1_000_000:
        return f"{microseconds / 1_000_000:.2f} s"
    if microseconds >= 1000:
        return f"{microseconds / 1000:.2f} ms"
    return f"{microseconds:.0f} us"


def format_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _sanitize_filename(text: str) -> str:
    """Collapses a free-form string (e.g. a satellite name like "ISS
    (ZARYA)") into something safe as a directory/file name component on
    any common filesystem -- keeps alphanumerics, replaces everything
    else with underscores, and collapses repeats so "ISS (ZARYA)" becomes
    "ISS_ZARYA" rather than "ISS__ZARYA_"."""
    cleaned = "".join(c if c.isalnum() else "_" for c in text.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def fit_pgm_to_canvas(pgm: bytes, full_width: int, full_height: int, canvas: tk.Canvas) -> tk.PhotoImage:
    """Scales a PGM frame to fit inside canvas without cropping, distorting
    the aspect ratio, or ever combining a shrink with a subsequent magnify
    in the same render. Shared by TransitPanel's live preview and
    SerPlayerPanel's playback view -- both hit the exact same PhotoImage
    limitations (integer-only subsample/zoom) and the exact same quality
    trap if that constraint isn't respected, see the comment below."""
    image = tk.PhotoImage(data=pgm)
    canvas_w = canvas.winfo_width()
    canvas_h = canvas.winfo_height()
    if canvas_w <= 1 or canvas_h <= 1:
        # Canvas not laid out yet (e.g. the very first frame) -- show
        # at native size rather than divide by a meaningless 1x1 canvas.
        canvas_w, canvas_h = full_width, full_height

    # PhotoImage only supports integer subsample/zoom, and chaining both
    # in the same render (subsample to shrink, then zoom back up) was
    # tried and reverted -- it looks genuinely blurry/blocky: subsample
    # throws away most pixels (nearest-neighbor, no averaging), then zoom
    # replicates whatever's left into NxN blocks, compounding the loss
    # (reported: full-sensor ROI looked "flouté" while an ROI close to but
    # not exactly full size looked sharp -- that ROI just happened to need
    # zoom=1, no compounding). So: shrink OR magnify, never both, even if
    # that leaves an unused margin on one axis instead of filling the
    # canvas exactly.
    if full_width > canvas_w or full_height > canvas_h:
        factor = max(1, -(-full_width // canvas_w), -(-full_height // canvas_h))  # ceil
        if factor > 1:
            image = image.subsample(factor, factor)
    else:
        zoom = max(1, min(canvas_w // full_width, canvas_h // full_height, MAX_PREVIEW_ZOOM))
        if zoom > 1:
            image = image.zoom(zoom, zoom)
    return image


@dataclasses.dataclass
class CameraControlVars:
    """Exposure/gain Tk variables, shared between TransitPanel and
    JogWindow (see am5/gui/app.py) so their two independent slider widgets
    show and drive the exact same value -- moving one moves the other, via
    Tk's own built-in mechanism for multiple widgets bound to one Variable,
    rather than two copies that only agreed at connect time and silently
    drifted apart afterwards (the bug this was built to fix)."""

    exposure_log: tk.DoubleVar  # log10(microseconds); 3.0 = 1000us
    exposure_value: tk.StringVar  # formatted display text, e.g. "1.00 ms"
    gain: tk.DoubleVar
    gain_value: tk.StringVar

    @classmethod
    def create(cls) -> "CameraControlVars":
        exposure_log = tk.DoubleVar(value=3.0)
        exposure_value = tk.StringVar(value="1.00 ms")
        gain = tk.DoubleVar(value=300)
        gain_value = tk.StringVar(value="300")

        # One trace each, registered here rather than per-widget, so the
        # formatted label stays correct regardless of which of the two
        # sliders (or _configure_control_bounds's own .set() on connect)
        # changed the underlying value.
        exposure_log.trace_add("write", lambda *_args: exposure_value.set(format_exposure_us(10 ** exposure_log.get())))
        gain.trace_add("write", lambda *_args: gain_value.set(f"{gain.get():.0f}"))
        return cls(exposure_log, exposure_value, gain, gain_value)


@dataclasses.dataclass
class SiteVars:
    """Observer lat/lon/elevation, shared between ConnectionPanel and
    PassesPanel (see am5/gui/app.py) -- one place to enter it, used both to
    search for passes AND to tell the mount where it is (:SMTI#/:St#/:Sg#
    at connect time, see Mount.sync_site_and_time). Two separate copies
    used to exist -- PassesPanel's own fields, and ConnectionPanel silently
    defaulting to a hardcoded Geneva coordinate -- which meant the mount's
    own horizon/altitude-limit calculations (:GAT# codes 5/6) were computed
    for the wrong location unless the operator happened to be near Geneva.

    elevation_m is Passes-tab-only: the LX200 protocol this mount speaks
    has no wire command to set an altitude (only :St#/:Sg# for lat/lon --
    checked against the protocol doc), so it's never sent to the mount --
    it only feeds Skyfield's wgs84.latlon(..., elevation_m=...) for more
    accurate rise/set timing (a real horizon at altitude sits below the
    sea-level horizon Skyfield otherwise assumes by default)."""

    lat: tk.StringVar
    lon: tk.StringVar
    elevation_m: tk.StringVar

    @classmethod
    def create(cls) -> "SiteVars":
        return cls(lat=tk.StringVar(value="46.18"), lon=tk.StringVar(value="6.14"), elevation_m=tk.StringVar(value="0"))


# Hard ceiling on a single tracking run regardless of what duration_s comes
# out to -- starting well before rise is now allowed on purpose (operator
# preference: arm, start, let it wait), so this is what keeps an early
# start from commanding a stale rate for hours instead of just holding
# still until the pass actually begins (see Trajectory.interpolate's
# outside-the-window rate=0 clamp in am5/ephemeris.py).
MAX_TRACKING_DURATION_S = 20 * 60.0

# Simulate track's optional "random pointing error" training scenario
# (mock only): magnitude range per axis (arcmin), randomized independently
# for RA and DEC each run so the operator can't memorize a fixed
# correction. Tuned to this project's actual rig (1000mm F/4 + ASI290MC
# main: ~19.3'x10.9' FOV; SVBony 60mm F4 (240mm) + ASI678MM finder:
# ~110'x62' FOV, see logs/*/optics.txt) -- comfortably beyond the main
# camera's half-extent (~9.6'x5.5') so the ISS is never accidentally
# already in the main camera's frame, comfortably inside the finder's
# half-extent (~55'x31') so it's always still findable there without
# needing a blind search.
TRAINING_POINTING_ERROR_ARCMIN_RANGE = (12.0, 25.0)

# CalibrationPanel: calibration nudge, gentle and short (a static/steady target
# is assumed -- see the panel's own instructions -- so no ISS-motion
# contamination to worry about here, unlike a live pass). Rate/duration are
# only *defaults* -- editable in the UI, since how far a nudge moves the
# blob on screen also depends on the optical train's field of view: a long
# focal length (typical for ISS imaging) can have the blob leave a narrow
# frame entirely at these defaults, so the operator needs to be able to
# dial the nudge down.
GUIDING_CALIB_NUDGE_RATE_X = 5.0
GUIDING_CALIB_NUDGE_DURATION_S = 1.5
GUIDING_CALIB_SETTLE_S = 0.5
# Auto-guide correction: only fires when the detected offset exceeds this
# (avoids reacting to detection jitter) and no more often than this interval
# (a single arrow-key-equivalent pulse needs time to actually move the
# mount and show up in the next frame before judging whether more is needed).
GUIDING_DEADBAND_PX = 3.0
GUIDING_MIN_CORRECTION_INTERVAL_S = 1.0
GUIDING_PERP_PULSE_DURATION_S = 0.15
# CalibrationPanel's live preview/blob-detection frame cap -- bounds both
# detect_brightest_blob's cost and the preview tk.PhotoImage's size
# regardless of the main camera's actual sensor resolution, see
# CalibrationPanel.handle_camera_event's own comment for the incident
# this fixes (a high-res main camera froze the whole window).
MAX_CALIBRATION_PREVIEW_DIM = 480


class ConnectionPanel(ttk.Frame):
    """Both device connections in one place -- mount and camera are two
    independent USB devices/workers, but there's no reason an operator
    should have to visit two different tabs just to plug both in before
    doing anything else."""

    # Default optics for the mock cameras' simulated star field/ISS size --
    # editable per-session via the sliders below, unlocked only in Mock
    # mode (a real camera reports its own sensor size; focal length is a
    # property of whichever telescope/lens is actually mounted, not
    # something software can query). Plate scale = 206265 * pixel_size_um
    # / (1000 * focal_length_mm) arcsec/px.
    #
    # Main: ASI290MC (1936x1096, 2.9µm pixels) on the main 1000mm tube.
    MAIN_DEFAULT_FOCAL_MM = 1000
    MAIN_DEFAULT_SENSOR_W = 1936
    MAIN_DEFAULT_SENSOR_H = 1096
    MAIN_DEFAULT_PIXEL_UM = 2.9
    # Finder: ASI678MM (3840x2160, 2.0µm pixels) on an SVBony 60mm F/4
    # (240mm focal length) finder scope.
    FINDER_DEFAULT_FOCAL_MM = 240
    FINDER_DEFAULT_SENSOR_W = 3840
    FINDER_DEFAULT_SENSOR_H = 2160
    FINDER_DEFAULT_PIXEL_UM = 2.0

    @staticmethod
    def _plate_scale_arcsec_per_px(focal_length_mm: float, pixel_size_um: float) -> float:
        return 206265.0 * pixel_size_um / (1000.0 * focal_length_mm)

    def __init__(
        self, parent: tk.Misc, mount_worker: MountWorker, camera_worker: CameraWorker,
        on_connection_change: Callable[[bool], None],
        get_optical_train: Callable[[], OpticalTrain | None] | None = None,
        site_vars: SiteVars | None = None,
        map_widget_cls: type = TkinterMapView,
        finder_worker: CameraWorker | None = None,
        finder_state: FinderState | None = None,
    ):
        super().__init__(parent, padding=10)
        self._worker = mount_worker
        self._camera_worker = camera_worker
        self._finder_worker = finder_worker
        # Shared with FinderCameraPanel/FinderWindow/TransitPanel (same
        # instance, owned by App) -- so the plate scale each camera is
        # ACTUALLY connected with lands where calibration/correction read
        # it (see calibrate_from_frames/get_correction_arcsec's callers),
        # instead of a separately-typed, easily-stale duplicate value.
        self._finder_state = finder_state
        self._on_connection_change = on_connection_change
        # Real configured plate scale for the mock camera's simulated star
        # field/ISS size -- optional so this panel still works without an
        # ExposurePanel wired in (tests). See TransitPanel's own copy of
        # this same reasoning (get_optical_train's caller used to live
        # there, before camera connection moved to this tab).
        self._get_optical_train = get_optical_train
        # Shared with PassesPanel (same instance, owned by App) when passed
        # -- see SiteVars' docstring for the bug this fixes.
        self._site_vars = site_vars if site_vars is not None else SiteVars.create()
        self._geocode_results: "queue.Queue[tuple[str, object]]" = queue.Queue()

        columns = ttk.Frame(self)
        columns.pack(fill="both", expand=True)
        left = ttk.Frame(columns)
        left.pack(side="left", fill="y", anchor="n")
        right = ttk.Frame(columns)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        mount_frame = ttk.LabelFrame(left, text="Mount", padding=8)
        mount_frame.pack(fill="x", anchor="n")

        self._kind_var = tk.StringVar(value="mock")
        self._address_var = tk.StringVar(value="/dev/ttyACM0")
        self._seed_var = tk.StringVar(value="")

        for i, (label, value) in enumerate([("Mock", "mock"), ("Serial", "serial"), ("TCP", "tcp")]):
            ttk.Radiobutton(mount_frame, text=label, variable=self._kind_var, value=value,
                             command=self._update_address_state).grid(row=i, column=0, sticky="w")

        self._address_entry = ttk.Entry(mount_frame, textvariable=self._address_var, width=24)
        self._address_entry.grid(row=1, column=1, sticky="w")
        ttk.Label(mount_frame, text="port / host:port").grid(row=2, column=1, sticky="w")

        ttk.Label(mount_frame, text="mock seed (optional)").grid(row=3, column=0, sticky="w")
        ttk.Entry(mount_frame, textvariable=self._seed_var, width=8).grid(row=3, column=1, sticky="w")

        ttk.Label(mount_frame, text="site lat").grid(row=4, column=0, sticky="w")
        ttk.Entry(mount_frame, textvariable=self._site_vars.lat, width=8).grid(row=4, column=1, sticky="w")
        ttk.Label(mount_frame, text="site lon").grid(row=5, column=0, sticky="w")
        ttk.Entry(mount_frame, textvariable=self._site_vars.lon, width=8).grid(row=5, column=1, sticky="w")
        ttk.Label(
            mount_frame, text="sent to the mount on connect -- also used by the Passes tab",
            foreground=PALETTE.fg_dim,
        ).grid(row=6, column=0, columnspan=2, sticky="w")

        ttk.Label(mount_frame, text="site elevation (m)").grid(row=7, column=0, sticky="w")
        ttk.Entry(mount_frame, textvariable=self._site_vars.elevation_m, width=8).grid(row=7, column=1, sticky="w")
        ttk.Label(
            mount_frame, text="Passes tab only -- the mount protocol has no altitude command",
            foreground=PALETTE.fg_dim,
        ).grid(row=8, column=0, columnspan=2, sticky="w")

        self._connect_button = ttk.Button(mount_frame, text="Connect", command=self._on_connect_click)
        self._connect_button.grid(row=9, column=0, pady=(8, 0))
        self._disconnect_button = ttk.Button(mount_frame, text="Disconnect", command=self._worker.disconnect, state="disabled")
        self._disconnect_button.grid(row=9, column=1, pady=(8, 0))

        self._status_var = tk.StringVar(value="Not connected")
        ttk.Label(mount_frame, textvariable=self._status_var).grid(row=10, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self._update_address_state()

        camera_frame = ttk.LabelFrame(left, text="Camera (ASI290MC, main tube)", padding=8)
        camera_frame.pack(fill="x", anchor="n", pady=(10, 0))

        self._camera_kind_var = tk.StringVar(value="mock")
        ttk.Radiobutton(
            camera_frame, text="Mock", variable=self._camera_kind_var, value="mock",
            command=lambda: self._update_mock_optics_state("main"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            camera_frame, text="Real ASI camera", variable=self._camera_kind_var, value="real",
            command=lambda: self._update_mock_optics_state("main"),
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(camera_frame, text="camera id").grid(row=1, column=0, sticky="w")
        self._camera_id_var = tk.StringVar(value="0")
        ttk.Entry(camera_frame, textvariable=self._camera_id_var, width=6).grid(row=1, column=1, sticky="w")

        (
            self._main_focal_var, self._main_sensor_w_var, self._main_sensor_h_var,
            self._main_pixel_var, self._main_scale_label_var,
        ) = self._build_mock_optics_rows(
            camera_frame, start_row=2, prefix="main",
            focal_mm=self.MAIN_DEFAULT_FOCAL_MM, sensor_w=self.MAIN_DEFAULT_SENSOR_W,
            sensor_h=self.MAIN_DEFAULT_SENSOR_H, pixel_um=self.MAIN_DEFAULT_PIXEL_UM,
        )

        self._camera_connect_button = ttk.Button(camera_frame, text="Connect", command=self._on_camera_connect_click)
        self._camera_connect_button.grid(row=7, column=0, pady=(6, 0))
        self._camera_disconnect_button = ttk.Button(
            camera_frame, text="Disconnect", command=self._camera_worker.disconnect, state="disabled",
        )
        self._camera_disconnect_button.grid(row=7, column=1, pady=(6, 0))
        self._camera_status_var = tk.StringVar(value="Not connected")
        ttk.Label(camera_frame, textvariable=self._camera_status_var).grid(row=8, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._update_mock_optics_state("main")

        # Finder scope camera -- entirely optional, greyed out if no
        # finder_worker was passed in (e.g. tests, or a build without the
        # finder feature wired up).
        finder_frame = ttk.LabelFrame(left, text="Finder camera (ASI678MM, optional)", padding=8)
        finder_frame.pack(fill="x", anchor="n", pady=(10, 0))
        self._finder_kind_var = tk.StringVar(value="mock")
        finder_mock_radio = ttk.Radiobutton(
            finder_frame, text="Mock", variable=self._finder_kind_var, value="mock",
            command=lambda: self._update_mock_optics_state("finder"),
        )
        finder_mock_radio.grid(row=0, column=0, sticky="w")
        finder_real_radio = ttk.Radiobutton(
            finder_frame, text="Real ASI camera", variable=self._finder_kind_var, value="real",
            command=lambda: self._update_mock_optics_state("finder"),
        )
        finder_real_radio.grid(row=0, column=1, sticky="w")
        ttk.Label(finder_frame, text="camera id").grid(row=1, column=0, sticky="w")
        self._finder_id_var = tk.StringVar(value="1")
        finder_id_entry = ttk.Entry(finder_frame, textvariable=self._finder_id_var, width=6)
        finder_id_entry.grid(row=1, column=1, sticky="w")

        (
            self._finder_focal_var, self._finder_sensor_w_var, self._finder_sensor_h_var,
            self._finder_pixel_var, self._finder_scale_label_var,
        ) = self._build_mock_optics_rows(
            finder_frame, start_row=2, prefix="finder",
            focal_mm=self.FINDER_DEFAULT_FOCAL_MM, sensor_w=self.FINDER_DEFAULT_SENSOR_W,
            sensor_h=self.FINDER_DEFAULT_SENSOR_H, pixel_um=self.FINDER_DEFAULT_PIXEL_UM,
            focal_pixel_always_editable=True,
        )

        self._finder_connect_button = ttk.Button(finder_frame, text="Connect", command=self._on_finder_connect_click)
        self._finder_connect_button.grid(row=7, column=0, pady=(6, 0))
        self._finder_disconnect_button = ttk.Button(
            finder_frame, text="Disconnect",
            command=self._finder_worker.disconnect if self._finder_worker is not None else (lambda: None),
            state="disabled",
        )
        self._finder_disconnect_button.grid(row=7, column=1, pady=(6, 0))
        self._finder_status_var = tk.StringVar(value="Not connected")
        ttk.Label(finder_frame, textvariable=self._finder_status_var).grid(row=8, column=0, columnspan=2, sticky="w", pady=(4, 0))
        if self._finder_worker is None:
            for w in (finder_mock_radio, finder_real_radio, finder_id_entry, self._finder_connect_button):
                w.configure(state="disabled")
            self._finder_status_var.set("Not available (no finder worker configured)")
        self._update_mock_optics_state("finder")

        location_frame = ttk.LabelFrame(right, text="Location", padding=8)
        location_frame.pack(fill="both", expand=True)
        search_row = ttk.Frame(location_frame)
        search_row.pack(fill="x")
        ttk.Label(search_row, text="search city/town").pack(side="left")
        self._city_search_var = tk.StringVar(value="")
        city_entry = ttk.Entry(search_row, textvariable=self._city_search_var, width=28)
        city_entry.pack(side="left", padx=(4, 4))
        city_entry.bind("<Return>", lambda _e: self._on_city_search_click())
        self._city_search_button = ttk.Button(search_row, text="Search", command=self._on_city_search_click)
        self._city_search_button.pack(side="left")
        self._map_status_var = tk.StringVar(value="Click the map, or search a city/town, to set the site above.")
        ttk.Label(location_frame, textvariable=self._map_status_var, foreground=PALETTE.fg_dim, wraplength=380).pack(
            anchor="w", pady=(4, 4)
        )

        # map_widget_cls: real TkinterMapView by default; tests pass a
        # lightweight stub instead -- the real widget starts ~26 daemon
        # threads per instance (tile pre-cache + a load pool) that don't
        # stop until the Tk root is destroyed and can still be mid
        # network-request at process exit, which segfaults the interpreter
        # during teardown once enough of them pile up across a test run
        # constructing many ConnectionPanels.
        self._map_widget = map_widget_cls(location_frame, width=380, height=280, corner_radius=0)
        self._map_widget.pack(fill="both", expand=True)
        try:
            start_lat, start_lon = float(self._site_vars.lat.get()), float(self._site_vars.lon.get())
        except ValueError:
            start_lat, start_lon = 46.18, 6.14
        self._map_widget.set_position(start_lat, start_lon, marker=True)
        self._map_widget.set_zoom(6)
        self._map_widget.add_left_click_map_command(self._on_map_click)

        self.after(200, self._poll_geocode_results)

    def _build_mock_optics_rows(
        self, parent: tk.Misc, start_row: int, prefix: str,
        focal_mm: float, sensor_w: int, sensor_h: int, pixel_um: float,
        focal_pixel_always_editable: bool = False,
    ) -> tuple[tk.StringVar, tk.StringVar, tk.StringVar, tk.StringVar, tk.StringVar]:
        """Focal length / sensor resolution / pixel size fields for a mock
        camera's simulated optics -- sensor W/H are only meaningful (and
        only editable) in Mock mode, since a real camera reports its own
        sensor size (see the "connected" event's width/height). Unlocked/
        locked by _update_mock_optics_state. Returns the four StringVars
        plus a live plate-scale readout var.

        focal_pixel_always_editable: if True, focal length and pixel size
        stay editable in Real mode too (only sensor W/H get locked) --
        neither is auto-reported by ANY camera, real or mock, unlike
        sensor size. The main camera doesn't need this (its real-mode
        plate scale comes from the Exposure calc tab's own fields
        instead, see ConnectionPanel._on_camera_connect_click), but the
        finder has no other source for these two values in real mode --
        confirmed missing entirely before this parameter existed, which
        silently left FinderState.finder_plate_scale_arcsec stuck at its
        1.0 default for any real finder camera, corrupting both the FOV
        rectangle's size and every finder-based correction's magnitude."""
        focal_var = tk.StringVar(value=str(focal_mm))
        sensor_w_var = tk.StringVar(value=str(sensor_w))
        sensor_h_var = tk.StringVar(value=str(sensor_h))
        pixel_var = tk.StringVar(value=str(pixel_um))
        scale_label_var = tk.StringVar(value="")

        ttk.Label(parent, text="focal length (mm)").grid(row=start_row, column=0, sticky="w")
        focal_entry = ttk.Entry(parent, textvariable=focal_var, width=8)
        focal_entry.grid(row=start_row, column=1, sticky="w")
        ttk.Label(parent, text="sensor W x H (px)").grid(row=start_row + 1, column=0, sticky="w")
        sensor_frame = ttk.Frame(parent)
        sensor_frame.grid(row=start_row + 1, column=1, sticky="w")
        sensor_w_entry = ttk.Entry(sensor_frame, textvariable=sensor_w_var, width=6)
        sensor_w_entry.pack(side="left")
        ttk.Label(sensor_frame, text="x").pack(side="left")
        sensor_h_entry = ttk.Entry(sensor_frame, textvariable=sensor_h_var, width=6)
        sensor_h_entry.pack(side="left")
        ttk.Label(parent, text="pixel size (µm)").grid(row=start_row + 2, column=0, sticky="w")
        pixel_entry = ttk.Entry(parent, textvariable=pixel_var, width=8)
        pixel_entry.grid(row=start_row + 2, column=1, sticky="w")
        ttk.Label(parent, textvariable=scale_label_var, foreground=PALETTE.fg_dim).grid(
            row=start_row + 3, column=0, columnspan=2, sticky="w", pady=(2, 4),
        )

        def _update_scale_label(*_args: object) -> None:
            try:
                scale = self._plate_scale_arcsec_per_px(float(focal_var.get()), float(pixel_var.get()))
                scale_label_var.set(f"→ {scale:.3f} arcsec/px")
            except (ValueError, ZeroDivisionError):
                scale_label_var.set("→ invalid focal length / pixel size")

        for var in (focal_var, pixel_var):
            var.trace_add("write", _update_scale_label)
        _update_scale_label()

        # Store widgets directly (not collected after the fact) so
        # _update_mock_optics_state can lock/unlock exactly these.
        mock_only_widgets = [sensor_w_entry, sensor_h_entry] if focal_pixel_always_editable else [
            focal_entry, sensor_w_entry, sensor_h_entry, pixel_entry,
        ]
        setattr(self, f"_{prefix}_optics_widgets", mock_only_widgets)
        if focal_pixel_always_editable:
            setattr(self, f"_{prefix}_optics_always_editable_widgets", [focal_entry, pixel_entry])

        return focal_var, sensor_w_var, sensor_h_var, pixel_var, scale_label_var

    def _update_mock_optics_state(self, which: str) -> None:
        """Locks the sensor-size fields (and, for the main camera, also
        focal/pixel -- see _build_mock_optics_rows' focal_pixel_always_
        editable param) for `which` ("main" or "finder") unless that
        device's kind var is set to Mock."""
        kind_var = self._camera_kind_var if which == "main" else self._finder_kind_var
        is_mock = kind_var.get() == "mock"
        widgets = getattr(self, f"_{which}_optics_widgets", [])
        for w in widgets:
            w.configure(state="normal" if is_mock else "disabled")

    def _update_address_state(self) -> None:
        self._address_entry.configure(state="disabled" if self._kind_var.get() == "mock" else "normal")

    def _on_connect_click(self) -> None:
        seed_text = self._seed_var.get().strip()
        seed = int(seed_text) if seed_text else None
        try:
            latitude_deg, longitude_deg = float(self._site_vars.lat.get()), float(self._site_vars.lon.get())
        except ValueError:
            self._status_var.set("Invalid site lat/lon")
            return
        self._connect_button.configure(state="disabled")
        self._status_var.set("Connecting...")
        self._worker.connect(
            self._kind_var.get(), address=self._address_var.get(), mock_seed=seed,
            latitude_deg=latitude_deg, longitude_deg=longitude_deg,
        )

    # -- location: map click / city search -------------------------------------

    def _apply_location(self, lat: float, lon: float, label: str | None = None) -> None:
        self._site_vars.lat.set(f"{lat:.4f}")
        self._site_vars.lon.set(f"{lon:.4f}")
        self._map_widget.delete_all_marker()
        self._map_widget.set_position(lat, lon, marker=True)
        self._map_widget.set_zoom(9)
        self._map_status_var.set(label or f"{lat:.4f}, {lon:.4f}")

    def _on_map_click(self, coords: tuple[float, float]) -> None:
        lat, lon = coords
        self._apply_location(lat, lon)

    def _on_city_search_click(self) -> None:
        query = self._city_search_var.get().strip()
        if not query:
            return
        self._city_search_button.configure(state="disabled")
        self._map_status_var.set(f"Searching for {query!r}...")
        threading.Thread(target=self._geocode_city, args=(query,), daemon=True).start()

    def _geocode_city(self, query: str) -> None:
        # Off the Tk thread -- geopy's Nominatim call is a blocking network
        # request, same reasoning as PassesPanel's own background fetch
        # thread (see _fetch_and_find). tkintermapview has its own
        # set_address() that does this, but it mixes the network call with
        # Tk widget updates internally, which isn't safe to run off-thread
        # -- so this does the geocoding itself and only touches the map
        # from _poll_geocode_results, back on the Tk thread.
        try:
            geolocator = Nominatim(user_agent="am5-iss-tracker")
            location = geolocator.geocode(query, timeout=10)
        except (GeocoderTimedOut, GeocoderUnavailable) as exc:
            self._geocode_results.put(("error", str(exc)))
            return
        except Exception as exc:  # noqa: BLE001 - surfaced to the panel, not fatal
            self._geocode_results.put(("error", str(exc)))
            return
        if location is None:
            self._geocode_results.put(("not_found", query))
            return
        self._geocode_results.put(("found", (location.latitude, location.longitude, location.address)))

    def _poll_geocode_results(self) -> None:
        try:
            while True:
                kind, payload = self._geocode_results.get_nowait()
                self._city_search_button.configure(state="normal")
                if kind == "found":
                    lat, lon, address = payload
                    self._apply_location(lat, lon, label=address)
                elif kind == "not_found":
                    self._map_status_var.set(f"No results for {payload!r}")
                elif kind == "error":
                    self._map_status_var.set(f"Search failed: {payload}")
        except queue.Empty:
            pass
        self.after(200, self._poll_geocode_results)

    def handle_event(self, event: WorkerEvent) -> None:
        if event.kind == "connected":
            self._status_var.set(f"Connected — firmware {event.payload['firmware']}")
            self._disconnect_button.configure(state="normal")
            self._on_connection_change(True)
        elif event.kind == "connect_error":
            self._status_var.set(f"Connection failed: {event.payload['message']}")
            self._connect_button.configure(state="normal")
        elif event.kind == "disconnected":
            self._status_var.set("Not connected")
            self._connect_button.configure(state="normal")
            self._disconnect_button.configure(state="disabled")
            self._on_connection_change(False)

    def _on_camera_connect_click(self) -> None:
        try:
            camera_id = int(self._camera_id_var.get())
        except ValueError:
            self._camera_status_var.set("Invalid camera id")
            return
        self._camera_connect_button.configure(state="disabled")
        self._camera_status_var.set("Connecting...")
        kind = self._camera_kind_var.get()
        if kind == "mock":
            # Mock mode: this panel's own focal/sensor/pixel fields are the
            # single source of truth -- NOT the Exposure calc tab's optical
            # train, which models a hypothetical setup for exposure planning
            # and isn't necessarily what the mock should simulate right now.
            try:
                focal_mm = float(self._main_focal_var.get())
                sensor_w = int(float(self._main_sensor_w_var.get()))
                sensor_h = int(float(self._main_sensor_h_var.get()))
                pixel_um = float(self._main_pixel_var.get())
            except ValueError:
                self._camera_status_var.set("Invalid focal length / sensor / pixel size")
                self._camera_connect_button.configure(state="normal")
                return
            plate_scale = self._plate_scale_arcsec_per_px(focal_mm, pixel_um)
            self._camera_worker.connect(
                kind, camera_id=camera_id, plate_scale_arcsec_per_px=plate_scale,
                mock_sensor_width=sensor_w, mock_sensor_height=sensor_h,
            )
            if self._finder_state is not None:
                self._finder_state.main_plate_scale_arcsec = plate_scale
            return
        plate_scale = None
        if self._get_optical_train is not None:
            train = self._get_optical_train()
            if train is not None:
                plate_scale = train.plate_scale_arcsec_per_px
        self._camera_worker.connect(kind, camera_id=camera_id, plate_scale_arcsec_per_px=plate_scale)
        if self._finder_state is not None and plate_scale is not None:
            self._finder_state.main_plate_scale_arcsec = plate_scale

    def handle_camera_event(self, event: CameraEvent) -> None:
        if event.kind == "connected":
            self._camera_status_var.set(
                f"Connected — {event.payload['width']}x{event.payload['height']}"
                f"{' colour' if event.payload['is_color'] else ' mono'}"
                f", {event.payload.get('bit_depth', 8)}-bit"
            )
            self._camera_disconnect_button.configure(state="normal")
        elif event.kind == "connect_error":
            self._camera_status_var.set(f"Connection failed: {event.payload['message']}")
            self._camera_connect_button.configure(state="normal")
        elif event.kind == "disconnected":
            self._camera_status_var.set("Not connected")
            self._camera_connect_button.configure(state="normal")
            self._camera_disconnect_button.configure(state="disabled")

    def _on_finder_connect_click(self) -> None:
        if self._finder_worker is None:
            return
        try:
            camera_id = int(self._finder_id_var.get())
        except ValueError:
            self._finder_status_var.set("Invalid camera id")
            return
        self._finder_connect_button.configure(state="disabled")
        self._finder_status_var.set("Connecting...")
        kind = self._finder_kind_var.get()
        if kind == "mock":
            try:
                focal_mm = float(self._finder_focal_var.get())
                sensor_w = int(float(self._finder_sensor_w_var.get()))
                sensor_h = int(float(self._finder_sensor_h_var.get()))
                pixel_um = float(self._finder_pixel_var.get())
            except ValueError:
                self._finder_status_var.set("Invalid focal length / sensor / pixel size")
                self._finder_connect_button.configure(state="normal")
                return
            plate_scale = self._plate_scale_arcsec_per_px(focal_mm, pixel_um)
            self._finder_worker.connect(
                kind, camera_id=camera_id, plate_scale_arcsec_per_px=plate_scale,
                mock_sensor_width=sensor_w, mock_sensor_height=sensor_h,
            )
            if self._finder_state is not None:
                self._finder_state.finder_plate_scale_arcsec = plate_scale
            return
        # Real camera: sensor size comes from the device itself (the
        # "connected" event), but focal length/pixel size are never
        # auto-reported by any camera -- read them from this panel's own
        # (always-editable in real mode, see focal_pixel_always_editable)
        # fields, same formula as the mock branch above. Without this, a
        # real finder camera left FinderState.finder_plate_scale_arcsec
        # stuck at its 1.0 default -- silently wrong for every finder
        # calibration and correction (confirmed missing entirely before
        # this fix, the main camera's equivalent case works only because
        # it has a different source, the Exposure calc tab's optical
        # train, see _on_camera_connect_click).
        try:
            focal_mm = float(self._finder_focal_var.get())
            pixel_um = float(self._finder_pixel_var.get())
            if self._finder_state is not None:
                self._finder_state.finder_plate_scale_arcsec = self._plate_scale_arcsec_per_px(focal_mm, pixel_um)
        except (ValueError, ZeroDivisionError):
            self._finder_status_var.set(
                "Warning: invalid finder focal length / pixel size -- calibration will use a wrong scale"
            )
        self._finder_worker.connect(kind, camera_id=camera_id)

    def handle_finder_camera_event(self, event: CameraEvent) -> None:
        if event.kind == "connected":
            self._finder_status_var.set(
                f"Connected — {event.payload['width']}x{event.payload['height']}"
                f"{' colour' if event.payload['is_color'] else ' mono'}"
                f", {event.payload.get('bit_depth', 8)}-bit"
            )
            self._finder_disconnect_button.configure(state="normal")
        elif event.kind == "connect_error":
            self._finder_status_var.set(f"Connection failed: {event.payload['message']}")
            self._finder_connect_button.configure(state="normal")
        elif event.kind == "disconnected":
            self._finder_status_var.set("Not connected")
            self._finder_connect_button.configure(state="normal")
            self._finder_disconnect_button.configure(state="disabled")


class SkyMapWidget:
    """Polar alt/az sky chart (N up, horizon at rim) with a constellation
    background -- shared by PassesPanel (planned track) and TransitPanel
    (planned track + live telescope position). Constellations/track are
    redrawn only when the pass changes (clear() + draw_*() + finish());
    the mount marker updates cheaply via set_data() on its own artist, like
    the along/cross-track error plot already does, so a live position tick
    doesn't require replotting everything else."""

    def __init__(self, parent: tk.Misc):
        self.figure = Figure(figsize=(4, 4), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="polar")
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self._mount_marker = None
        self._reset_axes()
        self.canvas.draw_idle()

    def widget(self) -> tk.Widget:
        return self.canvas.get_tk_widget()

    def _reset_axes(self) -> None:
        ax = self.ax
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 90)
        ax.set_rticks([0, 30, 60, 90])
        ax.set_yticklabels(["90°", "60°", "30°", "0°"])  # r=0 is zenith (alt 90), r=90 is horizon (alt 0)
        ax.set_rlabel_position(135)
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["N", "E", "S", "W"])
        style_axes(self.figure, ax)

    def clear(self) -> None:
        self.ax.clear()
        self._mount_marker = None  # the old artist was wiped along with the axes

    def draw_constellations(self, site: GeographicPosition, when) -> None:
        for shape in constellations_altaz(site, when):
            visible = {i for i, (_, alt) in enumerate(shape.stars_azalt) if alt >= 0}
            if not visible:
                continue
            for i, j in shape.lines:
                if i in visible and j in visible:
                    (az_i, alt_i), (az_j, alt_j) = shape.stars_azalt[i], shape.stars_azalt[j]
                    self.ax.plot(np.radians([az_i, az_j]), [90.0 - alt_i, 90.0 - alt_j], "-", color="0.55", linewidth=0.7, zorder=1)
            cen_az = sum(shape.stars_azalt[i][0] for i in visible) / len(visible)
            cen_alt = sum(shape.stars_azalt[i][1] for i in visible) / len(visible)
            self.ax.text(np.radians(cen_az), 90.0 - cen_alt, shape.name, fontsize=6, color="0.65", ha="center", zorder=1)

    def draw_track(self, az_deg: np.ndarray, alt_deg: np.ndarray, t_unix: np.ndarray, crossings: list) -> None:
        """az_deg/alt_deg/t_unix: same length, matching samples -- pass
        Trajectory.az_deg/alt_deg/t_unix directly for the real pass-time
        track, or am5.angles.equatorial_series_to_altaz's output (from
        Trajectory.ra_deg/dec_deg at some other reference time) to show
        "where this same track would be right now" instead."""
        az_rad = np.radians(az_deg)
        # zenith (alt 90) at center, horizon (alt 0) at rim -- clamped like
        # update_mount_marker's r=90-max(alt,0), so a below-horizon point
        # (only possible for the "rehearsal" -- recompute-at-now -- case;
        # the real pass-time track from Trajectory is always above horizon
        # by construction) sits at the rim instead of overflowing past it
        # and silently drifting out of sync with the telescope marker.
        r = 90.0 - np.maximum(alt_deg, 0.0)
        self.ax.plot(az_rad, r, "-", color="C0", linewidth=1.5, label="Track")
        self.ax.plot(az_rad[0], r[0], "o", color="green", markersize=7, label="Rise")
        self.ax.plot(az_rad[-1], r[-1], "s", color="red", markersize=7, label="Set")
        culm_idx = int(np.argmax(alt_deg))  # true max altitude, not the clamped r's argmin (ties at the rim otherwise)
        self.ax.plot(az_rad[culm_idx], r[culm_idx], "^", color="orange", markersize=8, label="Culminate")
        for i, crossing_t in enumerate(crossings):
            idx = int(np.argmin(np.abs(t_unix - crossing_t.timestamp())))
            self.ax.plot(az_rad[idx], r[idx], "x", color="purple", markersize=9, markeredgewidth=2,
                         label="Meridian" if i == 0 else None)

    def update_mount_marker(self, az_deg: float, alt_deg: float) -> None:
        """Cheap per-tick update -- creates the star marker on first call,
        just moves it (set_data) afterwards, no full redraw."""
        r = 90.0 - max(alt_deg, 0.0)
        theta = np.radians(az_deg)
        if self._mount_marker is None:
            (self._mount_marker,) = self.ax.plot(
                [theta], [r], "*", color=PALETTE.accent, markersize=16, markeredgecolor=PALETTE.fg, label="Telescope", zorder=5,
            )
            self._style_legend()
        else:
            self._mount_marker.set_data([theta], [r])
        self.canvas.draw_idle()

    def _style_legend(self) -> None:
        legend = self.ax.legend(loc="lower left", fontsize=7, framealpha=0.9, facecolor=PALETTE.bg_widget, edgecolor=PALETTE.border)
        for text in legend.get_texts():
            text.set_color(PALETTE.fg)

    def finish(self, legend: bool = True) -> None:
        self._reset_axes()
        if legend:
            self._style_legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()


class PassesPanel(ttk.Frame):
    def __init__(
        self, parent: tk.Misc, on_pass_selected: Callable[[Trajectory, PassWindow, list, GeographicPosition, str], None],
        site_vars: SiteVars | None = None,
    ):
        super().__init__(parent, padding=10)
        self._on_pass_selected = on_pass_selected
        self._results: "queue.Queue[tuple[str, object]]" = queue.Queue()
        # Shared with ConnectionPanel (same instance, owned by App) when
        # passed -- see SiteVars' docstring.
        self._site_vars = site_vars if site_vars is not None else SiteVars.create()
        self._satellite = None
        self._site = None
        self._passes: list[PassWindow] = []
        self._crossings_by_pass: dict[int, bool] = {}
        self._selected_iid: str | None = None

        columns_frame = ttk.Frame(self)
        columns_frame.pack(fill="both", expand=True)
        left = ttk.Frame(columns_frame)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(columns_frame)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        target_row = ttk.Frame(left)
        target_row.pack(fill="x", pady=(0, 6))
        ttk.Label(target_row, text="target").pack(side="left")
        target_names = list(KNOWN_SATELLITES) + [CUSTOM_SATELLITE_LABEL]
        self._target_var = tk.StringVar(value=target_names[0])
        ttk.Combobox(target_row, textvariable=self._target_var, values=target_names, state="readonly", width=20).pack(
            side="left", padx=(4, 10)
        )
        ttk.Label(target_row, text="custom NORAD ID").pack(side="left")
        self._custom_catnr_var = tk.StringVar(value="")
        self._custom_catnr_entry = ttk.Entry(target_row, textvariable=self._custom_catnr_var, width=8, state="disabled")
        self._custom_catnr_entry.pack(side="left", padx=(4, 0))
        self._target_var.trace_add("write", self._on_target_changed)

        form = ttk.Frame(left)
        form.pack(fill="x")
        self._horizon_var = tk.StringVar(value="10")
        self._lookahead_var = tk.StringVar(value="48")
        fields = [
            ("lat", self._site_vars.lat), ("lon", self._site_vars.lon), ("elevation m", self._site_vars.elevation_m),
            ("horizon", self._horizon_var), ("lookahead h", self._lookahead_var),
        ]
        for i, (label, var) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=0, column=i * 2, sticky="w", padx=(0 if i == 0 else 8, 0))
            ttk.Entry(form, textvariable=var, width=8).grid(row=0, column=i * 2 + 1, sticky="w")

        self._refresh_button = ttk.Button(left, text="Refresh passes", command=self._on_refresh_click)
        self._refresh_button.pack(anchor="w", pady=(8, 4))

        columns = ("rise", "culminate", "set", "max_el", "mag", "duration", "meridian")
        self._tree = ttk.Treeview(left, columns=columns, show="headings", height=8)
        headings = ["Rise (local)", "Culminate (local)", "Set (local)", "Max el (deg)", "Mag (est.)", "Duration (s)", "Meridian?"]
        for col, label in zip(columns, headings):
            self._tree.heading(col, text=label)
            self._tree.column(col, width=110, anchor="center")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_row_selected)

        self._detail_var = tk.StringVar(value="")
        # wraplength: the meridian-crossing detail line is long enough
        # unwrapped to blow out the whole window's width once a pass is
        # selected (confirmed -- it also squeezed the log bar at the
        # bottom of the window down to nothing, see app.py's packing
        # order comment for the other half of that fix).
        ttk.Label(left, textvariable=self._detail_var, justify="left", wraplength=820).pack(anchor="w", pady=(8, 0))

        ttk.Label(right, text="Sky track (N up, horizon at rim)").pack(anchor="w")
        self._sky_map = SkyMapWidget(right)
        self._sky_map.widget().pack(fill="both", expand=True)

        self.after(200, self._poll_results)

    def _draw_sky_map(self, trajectory: Trajectory, window: PassWindow, crossings: list) -> None:
        self._sky_map.clear()
        if self._site is not None:
            self._sky_map.draw_constellations(self._site, window.t_culminate)
        self._sky_map.draw_track(trajectory.az_deg, trajectory.alt_deg, trajectory.t_unix, crossings)
        self._sky_map.finish()

    def _on_target_changed(self, *_args: object) -> None:
        is_custom = self._target_var.get() == CUSTOM_SATELLITE_LABEL
        self._custom_catnr_entry.configure(state="normal" if is_custom else "disabled")

    def _resolve_target(self) -> tuple[int, float | None] | None:
        """(catnr, magnitude_ref), or None if a custom NORAD ID was
        selected but isn't a valid integer."""
        name = self._target_var.get()
        if name != CUSTOM_SATELLITE_LABEL:
            return KNOWN_SATELLITES[name]
        try:
            return int(self._custom_catnr_var.get()), None
        except ValueError:
            return None

    def _on_refresh_click(self) -> None:
        try:
            lat, lon = float(self._site_vars.lat.get()), float(self._site_vars.lon.get())
            elevation_m = float(self._site_vars.elevation_m.get())
            horizon, lookahead = float(self._horizon_var.get()), float(self._lookahead_var.get())
        except ValueError:
            self._detail_var.set("Invalid site/elevation/horizon/lookahead value")
            return
        target = self._resolve_target()
        if target is None:
            self._detail_var.set("Invalid custom NORAD ID")
            return
        catnr, magnitude_ref = target
        self._refresh_button.configure(state="disabled")
        self._detail_var.set("Fetching TLE and searching for passes...")
        threading.Thread(
            target=self._fetch_and_find, args=(lat, lon, elevation_m, horizon, lookahead, catnr, magnitude_ref), daemon=True,
        ).start()

    def _fetch_and_find(
        self, lat: float, lon: float, elevation_m: float, horizon: float, lookahead: float,
        catnr: int, magnitude_ref: float | None,
    ) -> None:
        try:
            cache_path = TLE_CACHE_DIR / f"tle_{catnr}.tle"  # per-satellite -- see load_satellite_tle's docstring
            satellite = load_satellite_tle(catnr, cache_path, max_age_hours=48.0)
            site = wgs84.latlon(lat, lon, elevation_m=elevation_m)
            passes = find_passes(satellite, site, horizon_deg=horizon, lookahead_hours=lookahead, magnitude_ref=magnitude_ref)
            self._results.put(("passes_ready", (satellite, site, passes)))
        except Exception as exc:  # noqa: BLE001 - surfaced to the panel, not fatal
            self._results.put(("error", str(exc)))

    def _on_row_selected(self, _event: object) -> None:
        selection = self._tree.selection()
        if not selection or self._satellite is None:
            return
        self._selected_iid = selection[0]
        window = self._passes[int(selection[0])]
        threading.Thread(target=self._compute_trajectory, args=(window,), daemon=True).start()

    def _compute_trajectory(self, window: PassWindow) -> None:
        try:
            trajectory = compute_trajectory(self._satellite, self._site, window.t_rise, window.t_set, step_s=0.05)
            crossings = meridian_crossings(trajectory)
            self._results.put(("trajectory_ready", (trajectory, window, crossings)))
        except Exception as exc:  # noqa: BLE001
            self._results.put(("error", str(exc)))

    def _poll_results(self) -> None:
        try:
            while True:
                kind, payload = self._results.get_nowait()
                if kind == "passes_ready":
                    self._satellite, self._site, self._passes = payload
                    self._populate_tree()
                    self._refresh_button.configure(state="normal")
                    self._detail_var.set(f"{len(self._passes)} pass(es) found — select one below")
                elif kind == "trajectory_ready":
                    trajectory, window, crossings = payload
                    self._show_trajectory_detail(trajectory, window, crossings)
                    self._draw_sky_map(trajectory, window, crossings)
                    if self._selected_iid is not None:
                        self._tree.set(self._selected_iid, "meridian", "Yes" if crossings else "No")
                    satellite_name = self._satellite.name if self._satellite is not None else ""
                    self._on_pass_selected(trajectory, window, crossings, self._site, satellite_name)
                elif kind == "error":
                    self._detail_var.set(f"Error: {payload}")
                    self._refresh_button.configure(state="normal")
        except queue.Empty:
            pass
        self.after(200, self._poll_results)

    def _populate_tree(self) -> None:
        self._tree.delete(*self._tree.get_children())
        self._selected_iid = None
        for i, window in enumerate(self._passes):
            duration = (window.t_set - window.t_rise).total_seconds()
            self._tree.insert("", "end", iid=str(i), values=(
                window.t_rise.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                window.t_culminate.astimezone().strftime("%H:%M:%S"),
                window.t_set.astimezone().strftime("%H:%M:%S"),
                f"{window.max_elevation_deg:.1f}",
                f"{window.magnitude_estimate:+.1f}" if not math.isnan(window.magnitude_estimate) else "N/A",
                f"{duration:.0f}",
                "?",
            ))

    def _show_trajectory_detail(self, trajectory: Trajectory, window: PassWindow, crossings: list) -> None:
        start_ra, start_dec, _, _ = trajectory.interpolate(float(trajectory.t_unix[0]))
        lines = [
            f"Rise {_local_and_utc(window.t_rise)}  --  Set {_local_and_utc(window.t_set)}",
            f"Start: RA={(start_ra % 360.0) / 15.0:.4f}h DEC={start_dec:+.4f} deg",
        ]
        lines.append(_meridian_detail_line(crossings, window))
        self._detail_var.set("\n".join(lines))


class TransitPanel(ttk.Frame):
    """The single screen used during an actual pass: mount tracking controls
    (left) and camera controls (right) in one view, so there's no tab
    switching mid-pass. Talks to both MountWorker and CameraWorker — two
    independent workers, so mount and camera events are handled through two
    separate methods (handle_mount_event/handle_camera_event) rather than
    one handler guessing which device an ambiguous event kind like
    "connected" came from.
    """

    def __init__(
        self, parent: tk.Misc, mount_worker: MountWorker, camera_worker: CameraWorker, out_dir: Path,
        live_offsets: LiveOffsets | None = None,
        axis_signs: AxisSigns | None = None, auto_guide_var: tk.BooleanVar | None = None,
        camera_vars: CameraControlVars | None = None,
        mount_lag_var: tk.DoubleVar | None = None, feedback_enabled_var: tk.BooleanVar | None = None,
        finder_state: FinderState | None = None,
    ):
        super().__init__(parent, padding=10)
        self._mount_worker = mount_worker
        self._camera_worker = camera_worker
        self._out_dir = out_dir
        # Shared with JogWindow (same instance, owned by App) when passed --
        # see CameraControlVars' docstring for why this fixes the two
        # sliders drifting apart.
        self._camera_vars = camera_vars if camera_vars is not None else CameraControlVars.create()

        # -- tracking state (mount side) --
        self._trajectory: Trajectory | None = None
        self._window: PassWindow | None = None
        self._site: GeographicPosition | None = None
        self._crossings: list = []
        self._satellite_name = ""
        self._capture_dir_prepared: Path | None = None  # see _prepare_capture_dir
        # Shared with JogWindow (same instance, owned by App) when passed --
        # set_axis_signs() below mutates it in place so both stay in sync
        # regardless of which one triggered a (re)calibration.
        self._axis_signs = axis_signs if axis_signs is not None else AxisSigns(ra=1.0, dec=1.0)
        # Shared with CalibrationPanel (same instance, owned by App) when the
        # caller passes one -- so a camera-detected correction lands in the
        # SAME offsets the tracking loop below is reading. Falls back to a
        # private instance so this panel still works standalone (tests,
        # or a build without the calibration tab wired in).
        self._offsets = live_offsets if live_offsets is not None else LiveOffsets()
        # Shared with CalibrationPanel (same instance, owned by App) when passed
        # -- CalibrationPanel reads this to decide whether to apply a detected
        # correction; the checkbox itself lives here since it's only useful
        # during an active pass (see set_auto_guide_available).
        self._auto_guide_var = auto_guide_var if auto_guide_var is not None else tk.BooleanVar(value=False)
        # Shared with CalibrationPanel (same instance, owned by App) when
        # passed -- CalibrationPanel's "Measure mount lag" writes here;
        # start/simulate below read it into TrackingConfig.mount_lag_s.
        # Falls back to a private var so this panel still works standalone.
        self._mount_lag_var = mount_lag_var if mount_lag_var is not None else tk.DoubleVar(value=0.0)
        self._feedback_enabled_var = feedback_enabled_var if feedback_enabled_var is not None else tk.BooleanVar(value=False)
        self._finder_state = finder_state
        self._armed = False
        self._mount_connected = False
        # Set from the "connected" WorkerEvent's own kind field (see
        # MountWorker._handle_connect), not the ConnectionPanel dropdown's
        # current selection -- reflects what's ACTUALLY connected right
        # now. Gates the training-scenario checkbox below; the worker
        # itself also refuses the injection outright if the connected
        # mount isn't mock (see MountWorker._handle_inject_training_
        # pointing_error), so this is UX only, not the safety boundary.
        self._mount_is_mock = False
        self._training_error_var = tk.BooleanVar(value=False)

        # -- camera state --
        self._camera_interactive_widgets: list[tk.Widget] = []
        # Subset of the above that must also grey out while recording is
        # active -- the worker refuses ROI/bit-depth changes mid-recording
        # (see CameraWorker._handle_set_roi/_handle_set_bit_depth, added
        # after a live ROI change was found to corrupt the SER file being
        # written), so disable them here too rather than let the operator
        # click a control that silently no-ops except for a log line.
        self._roi_bitdepth_widgets: list[tk.Widget] = []
        self._recording = False
        self._colour_id = 0
        self._is_color = False
        self._roi_x, self._roi_y, self._roi_w, self._roi_h = 0, 0, 640, 480
        self._sensor_width, self._sensor_height = 640, 480
        self._display_scale = 1
        self._display_w, self._display_h = 640, 480
        self._drag_start: tuple[int, int] | None = None
        self._drag_rect_id: int | None = None

        columns = ttk.Frame(self)
        columns.pack(fill="both", expand=True)
        left = ttk.Frame(columns)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(columns)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self._build_tracking_column(left)
        self._build_camera_column(right)

        # Arrow keys drive the same delta_t (↑ ↓) / perpendicular nudge
        # (← →) mechanisms as their buttons, from anywhere in this tab --
        # bound recursively on every widget, not just self/the preview
        # canvas: a binding on a container widget does NOT fire just
        # because some descendant happens to have focus (verified -- Tk
        # only consults the actually focused widget's own bindtags), so as
        # soon as the operator clicked ARM, an entry, anything besides the
        # canvas, keyboard control would silently stop doing anything at
        # all. Also flashes the matching button (see _on_perp_nudge_key /
        # _on_delta_t_key_press) so a keyboard-triggered action gets the
        # same visible feedback a mouse click already gets for free from
        # Tk's own button press animation. "break" pre-empts widgets with
        # their own arrow-key handling (the ROI entries' cursor move, the
        # exposure/gain sliders' value nudge).
        self._bind_offset_keys(self)

        self._set_camera_controls_enabled(False)
        self.after(300, self._poll_delta_t_display)

    # ==================================================================
    # Tracking column (mount side)
    # ==================================================================

    def _build_tracking_column(self, parent: tk.Misc) -> None:
        self._summary_var = tk.StringVar(value="No pass selected — pick one in the Passes tab")
        # wraplength: same long meridian-crossing line as PassesPanel's
        # _detail_var -- see that label's comment.
        ttk.Label(parent, textvariable=self._summary_var, justify="left", wraplength=620).pack(anchor="w")
        self._countdown_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self._countdown_var, foreground=PALETTE.accent_warn, font=("", 10, "bold")).pack(anchor="w")
        self._mount_radec_var = tk.StringVar(value="RA: --  DEC: --")
        ttk.Label(parent, textvariable=self._mount_radec_var, font=("", 11)).pack(anchor="w")

        button_row = ttk.Frame(parent)
        button_row.pack(anchor="w", pady=(8, 4))
        self._arm_button = ttk.Button(button_row, text="ARM", command=self._on_arm_click, state="disabled")
        self._arm_button.pack(side="left")
        self._start_button = ttk.Button(button_row, text="Start tracking", command=self._on_start_click, state="disabled")
        self._start_button.pack(side="left", padx=(4, 0))
        self._stop_button = ttk.Button(button_row, text="Stop tracking", command=self._mount_worker.stop_tracking, state="disabled")
        self._stop_button.pack(side="left", padx=(4, 0))
        self._simulate_button = ttk.Button(button_row, text="Simulate track", command=self._on_simulate_click, state="disabled")
        self._simulate_button.pack(side="left", padx=(4, 0))
        self._jog_goto_button = ttk.Button(button_row, text="GOTO (jog, keep pier side)", command=self._on_jog_goto_click, state="disabled")
        self._jog_goto_button.pack(side="left", padx=(4, 0))
        self._mount_goto_button = ttk.Button(button_row, text="GOTO (mount, auto pier side)", command=self._on_mount_goto_click, state="disabled")
        self._mount_goto_button.pack(side="left", padx=(4, 0))

        training_row = ttk.Frame(parent)
        training_row.pack(anchor="w", pady=(0, 4))
        self._training_error_check = ttk.Checkbutton(
            training_row, text="Simulate a random pointing error (mock only) -- rehearse finder-first acquisition",
            variable=self._training_error_var, state="disabled",
        )
        self._training_error_check.pack(side="left")

        offset_row = ttk.Frame(parent)
        offset_row.pack(anchor="w", pady=(4, 4))
        ttk.Label(offset_row, text="delta_t:").pack(side="left")
        ttk.Button(offset_row, text="-1s", width=4, command=lambda: self._offsets.adjust_delta_t(-1.0)).pack(side="left")
        self._delta_t_minus_button = ttk.Button(offset_row, text="-0.1s", width=5, command=lambda: self._offsets.adjust_delta_t(-0.1))
        self._delta_t_minus_button.pack(side="left")
        self._delta_t_var = tk.StringVar(value="+0.0s")
        ttk.Label(offset_row, textvariable=self._delta_t_var, width=8).pack(side="left")
        self._delta_t_plus_button = ttk.Button(offset_row, text="+0.1s", width=5, command=lambda: self._offsets.adjust_delta_t(0.1))
        self._delta_t_plus_button.pack(side="left")
        ttk.Button(offset_row, text="+1s", width=4, command=lambda: self._offsets.adjust_delta_t(1.0)).pack(side="left")

        perp_row = ttk.Frame(parent)
        perp_row.pack(anchor="w", pady=(0, 4))
        ttk.Label(perp_row, text="perpendicular nudge:").pack(side="left")
        self._perp_left_button = ttk.Button(perp_row, text="<", width=3, command=lambda: self._offsets.trigger_perp_pulse(-1.0))
        self._perp_left_button.pack(side="left")
        self._perp_right_button = ttk.Button(perp_row, text=">", width=3, command=lambda: self._offsets.trigger_perp_pulse(1.0))
        self._perp_right_button.pack(side="left")
        ttk.Label(parent, text="(↑ ↓ = delta_t, ← → = nudge -- from anywhere in this tab)",
                  foreground=PALETTE.fg_dim).pack(anchor="w", pady=(0, 4))

        self._build_track_legend(parent)

        self._auto_guide_check = ttk.Checkbutton(
            parent, text="Enable auto-guiding (camera-based cross-track correction)",
            variable=self._auto_guide_var, state="disabled",
        )
        self._auto_guide_check.pack(anchor="w")
        ttk.Label(
            parent, text="Needs calibration first, in the Calibration tab.",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(anchor="w", pady=(0, 4))

        self._finder_correct_var = tk.BooleanVar(value=False)
        self._finder_check = ttk.Checkbutton(
            parent, text="Enable finder correction (wide-field ISS blob → cross-track nudge)",
            variable=self._finder_correct_var, state="disabled",
        )
        self._finder_check.pack(anchor="w")
        ttk.Label(
            parent, text="Needs finder calibration first, in the Finder tab.",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(anchor="w", pady=(0, 4))

        self._feedback_check = ttk.Checkbutton(
            parent, text="Enable feedback trim (experimental PI on along/cross-track error)",
            variable=self._feedback_enabled_var,
        )
        self._feedback_check.pack(anchor="w")
        ttk.Label(
            parent, text="Conservative gains, clamped correction -- see the Calibration tab\n"
                         "for mount_lag_s (feedforward), independent of this.",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(anchor="w", pady=(0, 8))

        views = ttk.Notebook(parent)
        views.pack(fill="both", expand=True)
        error_tab = ttk.Frame(views)
        hist_tab = ttk.Frame(views)
        sky_tab = ttk.Frame(views)
        views.add(error_tab, text="Error plot")
        views.add(hist_tab, text="Histogram")
        views.add(sky_tab, text="Sky map")

        self._figure = Figure(figsize=(5, 3), dpi=100)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_xlabel("elapsed (s)")
        self._ax.set_ylabel("error (arcsec)")
        style_axes(self._figure, self._ax)
        (self._along_line,) = self._ax.plot([], [], label="along-track", color=PALETTE.accent)
        (self._cross_line,) = self._ax.plot([], [], label="cross-track", color=PALETTE.accent_warn)
        legend = self._ax.legend(loc="upper right", facecolor=PALETTE.bg_widget, edgecolor=PALETTE.border)
        for text in legend.get_texts():
            text.set_color(PALETTE.fg)
        self._canvas = FigureCanvasTkAgg(self._figure, master=error_tab)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._plot_t: list[float] = []
        self._plot_along: list[float] = []
        self._plot_cross: list[float] = []

        # Histogram: distribution of tracking errors -- useful for judging
        # overall pass quality (a tight, zero-centred peak = good; wide or
        # offset = systematic error) without having to read the time-series.
        self._hist_figure = Figure(figsize=(5, 3), dpi=100)
        self._hist_ax = self._hist_figure.add_subplot(111)
        self._hist_ax.set_xlabel("error (arcsec)")
        self._hist_ax.set_ylabel("frames")
        style_axes(self._hist_figure, self._hist_ax)
        self._hist_canvas = FigureCanvasTkAgg(self._hist_figure, master=hist_tab)
        self._hist_canvas.get_tk_widget().pack(fill="both", expand=True)

        self._sky_map = SkyMapWidget(sky_tab)
        self._sky_map.widget().pack(fill="both", expand=True)

    def _build_track_legend(self, parent: tk.Misc) -> None:
        """A small diagram, not just text -- "along-track" and
        "cross-track" name directions relative to the ISS's own motion
        through the frame, not the frame's fixed up/down/left/right, which
        is easy to get backwards from a text description alone. Colors
        match the error plot's along-track/cross-track lines (see
        _build_tracking_column) so the two views read as one system."""
        legend_frame = ttk.LabelFrame(parent, text="Legend: along-track vs. cross-track", padding=6)
        legend_frame.pack(anchor="w", pady=(0, 8), fill="x")

        canvas = tk.Canvas(legend_frame, width=260, height=90, background=PALETTE.bg_widget, highlightthickness=0)
        canvas.pack(side="left")
        cx, cy = 130, 45
        # along-track: horizontal, the ISS's own direction of travel through the frame
        canvas.create_line(20, cy, 240, cy, fill=PALETTE.accent, width=2, arrow=tk.LAST)
        canvas.create_text(cx, cy - 12, text="along-track", fill=PALETTE.accent, font=("", 9, "bold"))
        # cross-track: perpendicular to that, sideways drift off the track
        canvas.create_line(cx, 12, cx, 78, fill=PALETTE.accent_warn, width=2, arrow=tk.BOTH)
        canvas.create_text(cx + 46, 20, text="cross-track", fill=PALETTE.accent_warn, font=("", 9, "bold"))
        canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=PALETTE.fg, outline="")
        canvas.create_text(cx, cy + 14, text="ISS", fill=PALETTE.fg, font=("", 8))

        ttk.Label(
            legend_frame,
            text="Along-track: ahead/behind on its own path -- a timing\n"
                 "offset. Fixed with delta_t (↑ ↓ or the s buttons),\n"
                 "persistent until changed again.\n\n"
                 "Cross-track: off to the side of its path -- a pointing\n"
                 "offset. Fixed with the perpendicular nudge (← → or the\n"
                 "</> buttons), a short tap each time, or auto-guiding.",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(side="left", padx=(10, 0))

    def set_axis_signs(self, axis_signs: AxisSigns) -> None:
        # Mutate in place, don't reassign -- JogWindow may hold the same
        # shared instance (see App.__init__) and would otherwise keep
        # pointing at stale values.
        self._axis_signs.ra = axis_signs.ra
        self._axis_signs.dec = axis_signs.dec
        self._axis_signs.calibrated_pier_side = axis_signs.calibrated_pier_side

    def set_auto_guide_available(self, available: bool) -> None:
        self._auto_guide_check.configure(state="normal" if available else "disabled")
        if not available:
            self._auto_guide_var.set(False)

    def set_finder_correction_available(self, available: bool) -> None:
        self._finder_check.configure(state="normal" if available else "disabled")
        if not available:
            self._finder_correct_var.set(False)

    def set_mount_connected(self, connected: bool) -> None:
        self._mount_connected = connected
        if not connected:
            self._arm_button.configure(state="disabled")
            self._start_button.configure(state="disabled")
            self._simulate_button.configure(state="disabled")
            self._jog_goto_button.configure(state="disabled")
            self._mount_goto_button.configure(state="disabled")
        elif self._trajectory is not None:
            self._arm_button.configure(state="normal")
            self._simulate_button.configure(state="normal")
            self._jog_goto_button.configure(state="normal")
            self._mount_goto_button.configure(state="normal")

    def set_trajectory(
        self, trajectory: Trajectory, window: PassWindow, crossings: list, site: GeographicPosition,
        satellite_name: str = "",
    ) -> None:
        self._trajectory = trajectory
        self._window = window
        self._site = site
        self._crossings = crossings
        self._satellite_name = satellite_name
        duration = (window.t_set - window.t_rise).total_seconds()
        lines = [
            f"Rise {_local_and_utc(window.t_rise)}",
            f"Culminate {_local_and_utc(window.t_culminate)}",
            f"Set {_local_and_utc(window.t_set)}  ({duration:.0f}s, max el {window.max_elevation_deg:.1f} deg)",
        ]
        lines.append(_meridian_detail_line(crossings, window))
        self._summary_var.set("\n".join(lines))
        self._armed = False
        self._start_button.configure(state="disabled")
        if self._mount_connected:
            self._arm_button.configure(state="normal")
            self._simulate_button.configure(state="normal")
            self._jog_goto_button.configure(state="normal")
            self._mount_goto_button.configure(state="normal")

        self._redraw_sky_map()

    def _redraw_sky_map(self, rehearsal_now: datetime | None = None) -> None:
        """Without `rehearsal_now`: the real, future pass-time track (where
        the ISS will actually be during the real pass) -- correct for
        planning/live viewing once the real pass is underway.

        With `rehearsal_now`: the SAME RA/DEC track recomputed as it would
        appear right now, so it lines up with the live telescope marker
        (also "now"-based) -- otherwise a Manual GOTO/Simulate track done
        hours before the real pass points the mount at the correct RA/DEC,
        but the "Rise" marker (drawn for the real future rise time) and the
        telescope marker (drawn for actual now) land in different places on
        the chart even though the GOTO was correct -- Earth has rotated
        between the two reference times. See the incident that prompted
        this in am5/gui/panels.py's history."""
        if self._trajectory is None or self._window is None or self._site is None:
            return
        self._sky_map.clear()
        when = rehearsal_now or self._window.t_culminate
        self._sky_map.draw_constellations(self._site, when)
        if rehearsal_now is None:
            az_deg, alt_deg = self._trajectory.az_deg, self._trajectory.alt_deg
        else:
            az_deg, alt_deg = equatorial_series_to_altaz(
                self._trajectory.ra_deg % 360.0, self._trajectory.dec_deg,
                self._site.latitude.degrees, self._site.longitude.degrees, rehearsal_now,
            )
        self._sky_map.draw_track(az_deg, alt_deg, self._trajectory.t_unix, self._crossings)
        self._sky_map.finish()

    def _bind_offset_keys(self, widget: tk.Misc) -> None:
        widget.bind("<Left>", lambda _e: self._on_perp_nudge_key_press(-1.0))
        widget.bind("<Right>", lambda _e: self._on_perp_nudge_key_press(1.0))
        widget.bind("<Up>", lambda _e: self._on_delta_t_key_press(0.1))
        widget.bind("<Down>", lambda _e: self._on_delta_t_key_press(-0.1))
        for child in widget.winfo_children():
            self._bind_offset_keys(child)

    def _on_perp_nudge_key_press(self, sign: float) -> str:
        self._on_perp_nudge_key(sign)
        return "break"  # pre-empts the focused widget's own Left/Right handling (see _bind_offset_keys)

    def _on_perp_nudge_key(self, sign: float) -> None:
        self._offsets.trigger_perp_pulse(sign)
        self._flash_button(self._perp_left_button if sign < 0 else self._perp_right_button)

    def _on_delta_t_key_press(self, step: float) -> str:
        self._offsets.adjust_delta_t(step)
        self._flash_button(self._delta_t_plus_button if step > 0 else self._delta_t_minus_button)
        return "break"  # pre-empts the focused widget's own Up/Down handling (see _bind_offset_keys)

    def _flash_button(self, button: ttk.Button, duration_ms: int = int(GUIDING_PERP_PULSE_DURATION_S * 1000)) -> None:
        """Briefly shows a button as pressed -- for actions that fire a
        single short pulse (see trigger_perp_pulse) rather than a
        press-and-hold, so there's nothing for Tk's own button-press
        visual to attach to when triggered from the keyboard instead of an
        actual mouse click."""
        button.state(["pressed"])
        self.after(duration_ms, lambda: button.state(["!pressed"]))

    def _check_pass_timing(self) -> bool:
        """Only refuses a pass that has already set -- starting tracking
        against a window that's over can't do anything useful. Deliberately
        does NOT block starting early: the intended workflow is arm, then
        Start and let it wait/settle into the pass (duration_s is capped at
        MAX_TRACKING_DURATION_S regardless, so an early start can't spin
        for hours even though it's no longer refused outright)."""
        if self._window is None:
            return True
        now = datetime.now(timezone.utc)
        if now > self._window.t_set:
            messagebox.showerror(
                "Pass already over",
                f"This pass set at {_local_and_utc(self._window.t_set)} -- pick another pass.",
            )
            return False
        return True

    def _build_tracking_config(self) -> TrackingConfig:
        try:
            mount_lag_s = float(self._mount_lag_var.get())
        except (tk.TclError, ValueError):
            mount_lag_s = 0.0
        return TrackingConfig(mount_lag_s=mount_lag_s, enable_feedback=self._feedback_enabled_var.get())

    def _on_arm_click(self) -> None:
        self._armed = True
        self._start_button.configure(state="normal")

    def _on_start_click(self) -> None:
        if self._trajectory is None or self._window is None or not self._armed:
            return
        if not self._check_pass_timing():
            return
        self._redraw_sky_map()  # restore the real pass-time track, in case a rehearsal (Manual GOTO/Simulate) left a "now"-shifted view up
        self._plot_t.clear()
        self._plot_along.clear()
        self._plot_cross.clear()
        self._hist_ax.clear()
        self._hist_canvas.draw_idle()
        self._out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._out_dir / f"tracking_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"
        duration_s = max(0.0, (self._window.t_set - datetime.now(timezone.utc)).total_seconds())
        duration_s = min(duration_s, MAX_TRACKING_DURATION_S)
        self._mount_worker.start_tracking(
            self._trajectory, self._axis_signs, self._offsets, csv_path, duration_s, self._build_tracking_config(),
        )
        self._start_button.configure(state="disabled")
        self._stop_button.configure(state="normal")
        self._arm_button.configure(state="disabled")
        self._simulate_button.configure(state="disabled")
        self._jog_goto_button.configure(state="disabled")
        self._mount_goto_button.configure(state="disabled")

    def _on_simulate_click(self) -> None:
        """Replays the exact same real trajectory (RA/DEC/rates, including
        the meridian flip) starting right now instead of at the real rise
        time -- lets the operator watch the physical motion the pass will
        cause (tube/cable clearance) without waiting for the actual pass."""
        if self._trajectory is None or self._window is None:
            return
        self._redraw_sky_map(rehearsal_now=datetime.now(timezone.utc))
        now = datetime.now(timezone.utc).timestamp()
        shift_s = now - float(self._trajectory.t_unix[0])
        shifted = dataclasses.replace(self._trajectory, t_unix=self._trajectory.t_unix + shift_s)
        self._plot_t.clear()
        self._plot_along.clear()
        self._plot_cross.clear()
        self._out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._out_dir / f"simulate_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"
        duration_s = min(float(shifted.t_unix[-1]) - now, MAX_TRACKING_DURATION_S)
        if self._mount_is_mock and self._training_error_var.get():
            self._inject_training_pointing_error(shifted, now)
        self._mount_worker.start_tracking(
            shifted, self._axis_signs, self._offsets, csv_path, duration_s, self._build_tracking_config(),
        )
        self._simulate_button.configure(state="disabled")
        self._jog_goto_button.configure(state="disabled")
        self._mount_goto_button.configure(state="disabled")
        self._start_button.configure(state="disabled")
        self._arm_button.configure(state="disabled")
        self._stop_button.configure(state="normal")

    def _inject_training_pointing_error(self, shifted: Trajectory, now_unix: float) -> None:
        """Nudges the (mock) mount's own believed position by a random,
        realistic residual pointing error -- as if the last GOTO/sync
        landed a bit off, exactly the kind of thing a real operator has
        to notice and correct at the start of a real pass. Random sign
        and magnitude per axis, independently, each time -- see
        TRAINING_POINTING_ERROR_ARCMIN_RANGE's own comment for why the
        range is safe (always outside the main camera's narrow FOV,
        always inside the finder's wide one). RA bias divided by cos(dec)
        so the INJECTED ANGULAR offset on sky is what's actually in
        range, not the raw RA-degree number (which would read as a much
        bigger angle than intended away from the celestial equator)."""
        _ra_deg, dec_deg, _dra_dt, _ddec_dt = shifted.interpolate(now_unix)
        lo, hi = TRAINING_POINTING_ERROR_ARCMIN_RANGE
        dec_bias_arcmin = random.uniform(lo, hi) * random.choice((-1.0, 1.0))
        ra_bias_arcmin = random.uniform(lo, hi) * random.choice((-1.0, 1.0))
        dec_bias_deg = dec_bias_arcmin / 60.0
        ra_bias_deg = (ra_bias_arcmin / 60.0) / max(math.cos(math.radians(dec_deg)), 0.05)
        self._mount_worker.inject_training_pointing_error(ra_bias_deg, dec_bias_deg)

    def _goto_start_radec(self) -> tuple[float, float]:
        """RA/DEC where the ISS will be at pass start (or NOW if already
        in-progress) -- what to slew to before arming, so the mount is
        already pointed correctly when tracking begins rather than needing
        to catch up from wherever it happened to be."""
        now_unix = datetime.now(timezone.utc).timestamp()
        t_target = max(float(self._trajectory.t_unix[0]), now_unix)
        t_target = min(t_target, float(self._trajectory.t_unix[-1]))
        ra, dec, _, _ = self._trajectory.interpolate(t_target)
        return (ra % 360.0) / 15.0, dec

    def _disable_goto_buttons(self) -> None:
        """Disable everything that could start tracking while a GOTO runs --
        not just the GOTO button itself (confirmed on real hardware: Start/
        Simulate queued while jog_goto was still converging would inherit a
        large silent along-track error baked in at click time)."""
        self._jog_goto_button.configure(state="disabled")
        self._mount_goto_button.configure(state="disabled")
        self._arm_button.configure(state="disabled")
        self._start_button.configure(state="disabled")
        self._simulate_button.configure(state="disabled")

    def _on_jog_goto_click(self) -> None:
        """Jog-based GOTO to the pass-start position -- preserves pier side
        (never uses :MS#) but requires a valid axis-sign calibration."""
        if self._trajectory is None:
            return
        self._redraw_sky_map(rehearsal_now=datetime.now(timezone.utc))
        ra_hours, dec_deg = self._goto_start_radec()
        self._disable_goto_buttons()
        self._mount_worker.jog_goto(ra_hours, dec_deg, self._axis_signs)

    def _on_mount_goto_click(self) -> None:
        """Native mount GOTO (:MS#) to the pass-start position -- the
        firmware handles pier side automatically, so this works even
        without a valid axis-sign calibration. Trade-off: the mount
        chooses the pier side, which may differ from the current one."""
        if self._trajectory is None:
            return
        self._redraw_sky_map(rehearsal_now=datetime.now(timezone.utc))
        ra_hours, dec_deg = self._goto_start_radec()
        self._disable_goto_buttons()
        self._mount_worker.goto(ra_hours, dec_deg)

    def _poll_delta_t_display(self) -> None:
        dt, _ = self._offsets.snapshot()
        self._delta_t_var.set(f"{dt:+.1f}s")
        self._countdown_var.set(self._countdown_text())
        self.after(300, self._poll_delta_t_display)

    def _countdown_text(self) -> str:
        """Live, so a stale one-time snapshot can't sit there looking
        current for two hours while the operator waits for the wrong
        clock's "19:22" -- see am5/gui/panels.py's _local_and_utc docstring
        for the incident that prompted this."""
        if self._window is None:
            return ""
        now = datetime.now(timezone.utc)
        until_rise_s = (self._window.t_rise - now).total_seconds()
        until_set_s = (self._window.t_set - now).total_seconds()
        if until_rise_s > 0:
            if until_rise_s < 90:
                return f"Rise in {until_rise_s:.0f} s"
            return f"Rise in {until_rise_s / 60.0:.1f} min"
        if until_set_s > 0:
            if until_set_s < 90:
                return f"Pass in progress -- sets in {until_set_s:.0f} s"
            return f"Pass in progress -- sets in {until_set_s / 60.0:.1f} min"
        return f"Pass ended {-until_set_s / 60.0:.1f} min ago -- pick another pass"

    def _update_mount_marker(self, ra_hours: float, dec_deg: float) -> None:
        if self._site is None:
            return
        az_deg, alt_deg = equatorial_to_altaz(
            ra_hours * 15.0, dec_deg, self._site.latitude.degrees, self._site.longitude.degrees, datetime.now(timezone.utc),
        )
        self._sky_map.update_mount_marker(az_deg, alt_deg)

    def handle_mount_event(self, event: WorkerEvent) -> None:
        if event.kind == "connected":
            self._mount_is_mock = event.payload.get("connection_kind") == "mock"
            self._training_error_check.configure(state="normal" if self._mount_is_mock else "disabled")
            if not self._mount_is_mock:
                self._training_error_var.set(False)
        elif event.kind == "disconnected":
            self._mount_is_mock = False
            self._training_error_check.configure(state="disabled")
            self._training_error_var.set(False)
        elif event.kind == "position":
            side = event.payload.get("pier_side")
            side_text = f"  Pier side: {side}" if side else ""
            self._mount_radec_var.set(f"RA: {event.payload['ra_hours']:.4f}h  DEC: {event.payload['dec_deg']:+.4f} deg{side_text}")
            self._update_mount_marker(event.payload["ra_hours"], event.payload["dec_deg"])
        elif event.kind == "tracking_tick":
            actual_ra_deg, actual_dec_deg = event.payload["actual_ra_deg"], event.payload["actual_dec_deg"]
            if actual_ra_deg != "":  # only populated every error_log_every ticks, see tracker.py
                self._mount_radec_var.set(f"RA: {actual_ra_deg / 15.0:.4f}h  DEC: {actual_dec_deg:+.4f} deg")
                self._update_mount_marker(actual_ra_deg / 15.0, actual_dec_deg)
            self._plot_t.append(event.payload["elapsed_s"])
            self._plot_along.append(event.payload["along_track_arcsec"])
            self._plot_cross.append(event.payload["cross_track_arcsec"])
            self._along_line.set_data(self._plot_t, self._plot_along)
            self._cross_line.set_data(self._plot_t, self._plot_cross)
            self._ax.relim()
            self._ax.autoscale_view()
            self._canvas.draw_idle()
            self._maybe_apply_finder_correction()
            # Histogram: update every 10 ticks (no need to redraw every tick)
            if len(self._plot_along) % 10 == 0 and len(self._plot_along) > 0:
                self._hist_ax.clear()
                self._hist_ax.hist(self._plot_along, bins=20, alpha=0.7, color=PALETTE.accent, label="along-track")
                self._hist_ax.hist(self._plot_cross, bins=20, alpha=0.7, color=PALETTE.accent_warn, label="cross-track")
                self._hist_ax.axvline(0, color=PALETTE.fg_dim, linewidth=0.8, linestyle="--")
                self._hist_ax.set_xlabel("error (arcsec)")
                self._hist_ax.set_ylabel("frames")
                legend = self._hist_ax.legend(loc="upper right", facecolor=PALETTE.bg_widget, edgecolor=PALETTE.border)
                for text in legend.get_texts():
                    text.set_color(PALETTE.fg)
                style_axes(self._hist_figure, self._hist_ax)
                self._hist_canvas.draw_idle()
        elif event.kind == "jog_goto_result":
            if self._mount_connected and self._trajectory is not None:
                self._jog_goto_button.configure(state="normal")
                self._mount_goto_button.configure(state="normal")
                self._arm_button.configure(state="normal")
                self._simulate_button.configure(state="normal")
                if self._armed:
                    self._start_button.configure(state="normal")
        elif event.kind == "goto_result":
            # Native :MS# GOTO complete -- re-enable GOTO buttons
            if self._mount_connected and self._trajectory is not None:
                self._jog_goto_button.configure(state="normal")
                self._mount_goto_button.configure(state="normal")
                self._arm_button.configure(state="normal")
                self._simulate_button.configure(state="normal")
                if self._armed:
                    self._start_button.configure(state="normal")
        elif event.kind in ("tracking_stopped", "tracking_error"):
            self._stop_button.configure(state="disabled")
            if self._mount_connected and self._trajectory is not None:
                self._arm_button.configure(state="normal")
                self._simulate_button.configure(state="normal")
                self._jog_goto_button.configure(state="normal")
                self._mount_goto_button.configure(state="normal")

    def _maybe_apply_finder_correction(self) -> None:
        """Apply a cross-track correction from the finder camera if it has
        a blob locked AND its field has been calibrated against the main
        camera. Uses the same trigger_perp_pulse mechanism as auto-guiding
        -- gentle, bounded pulses, not instantaneous position jumps.
        Only active when the finder checkbox is checked.

        Regression fix: this method used to be defined on CalibrationPanel
        (which has neither self._finder_state nor self._finder_correct_var
        -- both are TransitPanel-only attributes, see __init__ above and
        the "Enable finder correction" checkbox in the camera column
        below), while its only call site was already correctly here, on
        TransitPanel.handle_mount_event's "tracking_tick" branch. Every
        real or simulated tracking session hit this the moment the first
        tracking_tick arrived (~1s in) -- an AttributeError there
        propagates out of App._pump_events BEFORE it reaches its own
        self.root.after(EVENT_POLL_MS, self._pump_events) reschedule call
        at the very end, permanently killing the whole event pump (not
        just this panel) for the rest of the session: no more camera
        previews, no more mount position updates, nothing -- matching a
        real, reported "the app just freezes" symptom far better than
        any per-frame performance cost does. Confirmed by reproducing an
        actual live Simulate-track session end-to-end against a mock
        rig and watching it crash on the first tracking_tick."""
        if self._finder_state is None or not self._finder_correct_var.get():
            return
        correction = self._finder_state.get_correction_arcsec()
        if correction is None:
            return
        _along_arcsec, cross_arcsec = correction
        if abs(cross_arcsec) < 5.0:  # ~5" dead-band -- don't over-correct noise
            return
        self._offsets.trigger_perp_pulse(1.0 if cross_arcsec > 0 else -1.0)

    # ==================================================================
    # Camera column
    # ==================================================================

    def _build_camera_column(self, parent: tk.Misc) -> None:
        # Connect/disconnect itself happens in the Connection tab (one place
        # for both devices) -- this is just a read-only echo of that state,
        # since it's still useful to see at a glance while working the ROI
        # and exposure controls below.
        self._camera_status_var = tk.StringVar(value="Not connected (connect in the Connection tab)")
        ttk.Label(parent, textvariable=self._camera_status_var).pack(anchor="w", pady=(0, 8))

        roi_frame = ttk.LabelFrame(parent, text="ROI — drag a rectangle on the preview, or type exact values", padding=8)
        roi_frame.pack(fill="x", pady=(8, 0))
        self._roi_x_var = tk.StringVar(value="0")
        self._roi_y_var = tk.StringVar(value="0")
        self._roi_w_var = tk.StringVar(value="640")
        self._roi_h_var = tk.StringVar(value="480")
        for i, (label, var) in enumerate([("x", self._roi_x_var), ("y", self._roi_y_var),
                                           ("w", self._roi_w_var), ("h", self._roi_h_var)]):
            ttk.Label(roi_frame, text=label).grid(row=0, column=i * 2, sticky="w")
            entry = ttk.Entry(roi_frame, textvariable=var, width=6)
            entry.grid(row=0, column=i * 2 + 1, sticky="w")
            self._camera_interactive_widgets.append(entry)
            self._roi_bitdepth_widgets.append(entry)
        roi_button = ttk.Button(roi_frame, text="Apply", command=self._on_apply_roi_entries)
        roi_button.grid(row=0, column=8, padx=(8, 0))
        self._camera_interactive_widgets.append(roi_button)
        self._roi_bitdepth_widgets.append(roi_button)
        reset_button = ttk.Button(roi_frame, text="Reset (full frame)", command=self._on_reset_roi)
        reset_button.grid(row=0, column=9, padx=(4, 0))
        self._camera_interactive_widgets.append(reset_button)
        self._roi_bitdepth_widgets.append(reset_button)

        controls_frame = ttk.LabelFrame(parent, text="Exposure / gain", padding=8)
        controls_frame.pack(fill="x", pady=(8, 0))
        controls_frame.columnconfigure(1, weight=1)

        ttk.Label(controls_frame, text="Exposure").grid(row=0, column=0, sticky="w")
        self._exposure_scale = ttk.Scale(
            controls_frame, from_=math.log10(32), to=math.log10(MAX_EXPOSURE_SLIDER_US),
            variable=self._camera_vars.exposure_log,
        )
        self._exposure_scale.grid(row=0, column=1, sticky="we", padx=(8, 8))
        self._exposure_scale.bind("<ButtonRelease-1>", self._on_exposure_release)
        self._camera_interactive_widgets.append(self._exposure_scale)
        ttk.Label(controls_frame, textvariable=self._camera_vars.exposure_value, width=10).grid(row=0, column=2, sticky="w")

        ttk.Label(controls_frame, text="Gain").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._gain_scale = ttk.Scale(controls_frame, from_=0, to=570, variable=self._camera_vars.gain)
        self._gain_scale.grid(row=1, column=1, sticky="we", padx=(8, 8), pady=(6, 0))
        self._gain_scale.bind("<ButtonRelease-1>", self._on_gain_release)
        self._camera_interactive_widgets.append(self._gain_scale)
        ttk.Label(controls_frame, textvariable=self._camera_vars.gain_value, width=10).grid(row=1, column=2, sticky="w", pady=(6, 0))

        preview_frame = ttk.LabelFrame(parent, text="Live preview (raw sensor, ~10Hz, not debayered) — drag to select ROI, click then use ← → to nudge", padding=8)
        preview_frame.pack(fill="both", expand=True, pady=(8, 0))
        self._preview_canvas = tk.Canvas(preview_frame, bg="black", highlightthickness=0, takefocus=True)
        self._preview_canvas.pack(fill="both", expand=True)
        self._preview_image: tk.PhotoImage | None = None  # keep a reference, Tk drops images without one
        self._preview_canvas_image_id: int | None = None
        self._preview_canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self._preview_canvas.bind("<B1-Motion>", self._on_drag_motion)
        self._preview_canvas.bind("<ButtonRelease-1>", self._on_drag_end)

        rec_frame = ttk.Frame(parent)
        rec_frame.pack(fill="x", pady=(8, 0))
        self._record_button = ttk.Button(rec_frame, text="Start recording (SER)", command=self._on_toggle_recording)
        self._record_button.pack(side="left")
        self._camera_interactive_widgets.append(self._record_button)
        self._snapshot_button = ttk.Button(rec_frame, text="Save FITS snapshot", command=self._on_snapshot_click)
        self._snapshot_button.pack(side="left", padx=(4, 0))
        self._camera_interactive_widgets.append(self._snapshot_button)
        ttk.Label(rec_frame, text="depth:").pack(side="left", padx=(8, 0))
        # Single control for both the SER recording AND the FITS snapshot --
        # no separate per-action bit depth, they always match (see
        # _on_bit_depth_selected). 16-bit costs roughly 2x the USB
        # bandwidth per frame (confirmed on a real ASI290MC: no fps hit at
        # 640x480/5ms where exposure time was already the bottleneck, but
        # expect a real hit at full sensor resolution or longer exposures).
        self._bit_depth_var = tk.StringVar(value="8")
        self._bit_depth_combo = ttk.Combobox(
            rec_frame, textvariable=self._bit_depth_var, values=("8", "16"), width=4,
            state="readonly",
        )
        self._bit_depth_combo.pack(side="left", padx=(4, 0))
        self._bit_depth_combo.bind("<<ComboboxSelected>>", self._on_bit_depth_selected)
        self._bit_depth_combo.configure(state="disabled")  # "readonly" once connected, not "normal" -- see _set_camera_controls_enabled
        self._roi_bitdepth_widgets.append(self._bit_depth_combo)

        self._stats_var = tk.StringVar(value="")
        self._stats_label = ttk.Label(parent, textvariable=self._stats_var)
        self._stats_label.pack(anchor="w", pady=(4, 0))
        buffer_row = ttk.Frame(parent)
        buffer_row.pack(fill="x", pady=(2, 0))
        ttk.Label(buffer_row, text="write buffer:").pack(side="left")
        self._buffer_var = tk.DoubleVar(value=0.0)
        self._buffer_bar = ttk.Progressbar(buffer_row, variable=self._buffer_var, maximum=100.0, length=140)
        self._buffer_bar.pack(side="left", padx=(4, 0))
        self._buffer_pct_var = tk.StringVar(value="")
        ttk.Label(buffer_row, textvariable=self._buffer_pct_var).pack(side="left", padx=(4, 0))
        self._file_size_var = tk.StringVar(value="")
        ttk.Label(buffer_row, textvariable=self._file_size_var).pack(side="left", padx=(10, 0))
        self._path_var = tk.StringVar(value=f"Output folder: {self._out_dir.resolve()}")
        self._path_label = ttk.Label(parent, textvariable=self._path_var, foreground=PALETTE.accent_ok)
        self._path_label.pack(anchor="w", pady=(2, 0))

    def _apply_roi(self, x: int, y: int, w: int, h: int) -> None:
        x = max(0, min(x, self._sensor_width - 1))
        y = max(0, min(y, self._sensor_height - 1))
        w = max(8, min(w, self._sensor_width - x))
        h = max(8, min(h, self._sensor_height - y))
        # Match the ASI SDK's own rounding (width multiple of 8, height
        # multiple of 2 -- see AsiCamera.set_roi) so the displayed X/Y/W/H
        # always reflects what was actually applied, not a value the
        # camera silently rounded down from.
        w = max(8, (w // 8) * 8)
        h = max(2, (h // 2) * 2)
        self._roi_x, self._roi_y, self._roi_w, self._roi_h = x, y, w, h
        self._roi_x_var.set(str(x))
        self._roi_y_var.set(str(y))
        self._roi_w_var.set(str(w))
        self._roi_h_var.set(str(h))
        self._camera_worker.set_roi(x, y, w, h)
        if self._finder_state is not None:
            # Feeds FinderState.main_fov_corners_px (see camera/finder.py)
            # so the finder preview's FOV rectangle shrinks to match a
            # smaller ROI instead of always claiming the full sensor's
            # field -- also called with the full sensor at connect time
            # (see handle_camera_event's "connected" branch above), so
            # this is the single place that keeps the rectangle in sync.
            self._finder_state.main_sensor_width = w
            self._finder_state.main_sensor_height = h
            self._finder_state.main_roi_offset_col = (x + w / 2.0) - self._sensor_width / 2.0
            self._finder_state.main_roi_offset_row = (y + h / 2.0) - self._sensor_height / 2.0

    def _on_apply_roi_entries(self) -> None:
        try:
            x, y = int(self._roi_x_var.get()), int(self._roi_y_var.get())
            w, h = int(self._roi_w_var.get()), int(self._roi_h_var.get())
        except ValueError:
            return
        self._apply_roi(x, y, w, h)

    def _on_reset_roi(self) -> None:
        self._apply_roi(0, 0, self._sensor_width, self._sensor_height)

    # -- ROI drag selection on the preview canvas ---------------------------

    def _clamp_to_display(self, x: int, y: int) -> tuple[int, int]:
        return max(0, min(x, self._display_w)), max(0, min(y, self._display_h))

    def _on_drag_start(self, event: tk.Event) -> None:
        self._preview_canvas.focus_set()  # dragging also grabs keyboard focus for arrow-key nudging
        if self._preview_canvas_image_id is None:
            return  # no frame yet -- nothing to select against
        x, y = self._clamp_to_display(event.x, event.y)
        self._drag_start = (x, y)
        if self._drag_rect_id is not None:
            self._preview_canvas.delete(self._drag_rect_id)
        self._drag_rect_id = self._preview_canvas.create_rectangle(x, y, x, y, outline="#00ff00", width=2)

    def _on_drag_motion(self, event: tk.Event) -> None:
        if self._drag_start is None or self._drag_rect_id is None:
            return
        x, y = self._clamp_to_display(event.x, event.y)
        x0, y0 = self._drag_start
        self._preview_canvas.coords(self._drag_rect_id, x0, y0, x, y)

    def _on_drag_end(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = self._clamp_to_display(event.x, event.y)
        self._drag_start = None
        if self._drag_rect_id is not None:
            self._preview_canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
            return  # accidental click, not a real drag -- ignore
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        scale = self._display_scale  # now possibly fractional (< 1 when the preview is magnified)
        self._apply_roi(
            round(self._roi_x + left * scale), round(self._roi_y + top * scale),
            round((right - left) * scale), round((bottom - top) * scale),
        )

    # -- exposure / gain sliders ---------------------------------------------
    # Live label updates are handled by CameraControlVars' own traces (fires
    # for both this panel's and JogWindow's sliders alike, whichever moved)
    # -- only the on-release commit to the camera is per-widget here.

    def _on_exposure_release(self, _event: tk.Event) -> None:
        self._camera_worker.set_exposure_us(round(10 ** self._camera_vars.exposure_log.get()))

    def _on_gain_release(self, _event: tk.Event) -> None:
        self._camera_worker.set_gain(round(self._camera_vars.gain.get()))

    def _on_toggle_recording(self) -> None:
        if self._recording:
            self._camera_worker.stop_recording()
        else:
            capture_dir = self._prepare_capture_dir()
            path = (capture_dir / f"capture_{datetime.now().strftime('%Y%m%dT%H%M%S')}.ser").resolve()
            self._write_capture_settings(path.with_suffix(".txt"))
            self._camera_worker.start_recording(path, observer="", instrument="ASI290MC", telescope="")

    def _on_snapshot_click(self) -> None:
        capture_dir = self._prepare_capture_dir()
        path = (capture_dir / f"snapshot_{datetime.now().strftime('%Y%m%dT%H%M%S')}.fits").resolve()
        self._camera_worker.save_fits_snapshot(path)

    def _prepare_capture_dir(self) -> Path:
        """Where the next recording/snapshot goes. With a pass selected
        (self._window set, see set_trajectory), everything from that pass
        -- every recording, every snapshot, plus pass_info.txt and
        skymap.png -- lands together in one dedicated subfolder, named
        from the pass identity (satellite + rise time) so re-arming or
        multiple recordings of the SAME pass land in the SAME folder
        rather than a fresh one per click. With no pass selected (e.g.
        just testing camera settings), falls back to the flat --out-dir
        behavior this app has always had."""
        if self._window is None:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            return self._out_dir
        capture_dir = self._out_dir / self._pass_folder_name()
        capture_dir.mkdir(parents=True, exist_ok=True)
        # pass_info.txt/skymap.png only need writing once per pass, not on
        # every recording/snapshot click -- _write_skymap in particular is
        # a synchronous matplotlib savefig() on the Tk main thread (no
        # worker involved, unlike camera/mount I/O), so redoing it on
        # every click risked a real, avoidable UI stutter right as the
        # operator starts recording during a live, time-critical pass.
        if self._capture_dir_prepared != capture_dir:
            self._write_pass_info(capture_dir)
            self._write_skymap(capture_dir)
            self._capture_dir_prepared = capture_dir
        return capture_dir

    def _pass_folder_name(self) -> str:
        assert self._window is not None
        name = _sanitize_filename(self._satellite_name) or "satellite"
        return f"{name}_{self._window.t_rise.strftime('%Y%m%dT%H%M%S')}"

    def _write_pass_info(self, capture_dir: Path) -> None:
        window = self._window
        assert window is not None
        lines = [
            f"Satellite: {self._satellite_name or '(unknown)'}",
            f"Rise:      {_local_and_utc(window.t_rise)}",
            f"Culminate: {_local_and_utc(window.t_culminate)}",
            f"Set:       {_local_and_utc(window.t_set)}",
            f"Duration:  {(window.t_set - window.t_rise).total_seconds():.0f} s",
            f"Max elevation: {window.max_elevation_deg:.1f} deg",
            f"Distance at culmination: {window.distance_km:.1f} km",
        ]
        if window.magnitude_estimate == window.magnitude_estimate:  # excludes NaN (no magnitude_ref was available)
            lines.append(f"Estimated magnitude: {window.magnitude_estimate:.1f}")
        lines.append(_meridian_detail_line(self._crossings, window))
        if self._site is not None:
            lines.append(
                f"Site: {self._site.latitude.degrees:.5f}, {self._site.longitude.degrees:.5f}, "
                f"{self._site.elevation.m:.0f} m"
            )
        try:
            (capture_dir / "pass_info.txt").write_text("\n".join(lines) + "\n")
        except OSError as exc:
            # Same "don't block the actual recording over a sidecar file"
            # reasoning as _write_skymap's own try/except right below --
            # this used to be uncaught, which would abort start_recording()
            # entirely (an unhandled exception in a Tk button callback)
            # over what should be a non-fatal write failure.
            self._emit_log_line(f"[warn] could not save pass_info.txt: {exc}")

    def _write_skymap(self, capture_dir: Path) -> None:
        # Reuses this panel's own live sky-map figure (already rendered by
        # set_trajectory/_redraw_sky_map for the currently selected pass)
        # rather than building a second one -- just saves it.
        try:
            self._sky_map.figure.savefig(capture_dir / "skymap.png", dpi=100)
        except Exception as exc:  # noqa: BLE001 - a plotting/IO hiccup here shouldn't block the actual recording
            self._emit_log_line(f"[warn] could not save skymap.png: {exc}")

    def _write_capture_settings(self, sidecar_path: Path) -> None:
        # FireCapture-style per-capture sidecar: the settings that were
        # actually in effect for THIS recording, not just whatever the
        # current live values happen to be later (gain/exposure/ROI can
        # change between recordings within the same pass).
        exposure_us = round(10 ** self._camera_vars.exposure_log.get())
        gain = round(self._camera_vars.gain.get())
        lines = [
            f"Time (UTC): {datetime.now(timezone.utc).isoformat()}",
            f"Exposure: {format_exposure_us(exposure_us)} ({exposure_us} us)",
            f"Gain: {gain}",
            f"ROI: {self._roi_w}x{self._roi_h} at ({self._roi_x},{self._roi_y}) of {self._sensor_width}x{self._sensor_height}",
            f"Bit depth: {self._bit_depth_var.get()}",
            f"Colour: {'colour' if self._is_color else 'mono'} (SER ColourID {self._colour_id})",
        ]
        if self._window is not None:
            lines.append(f"Pass: {self._satellite_name or '(unknown)'}, rise {_local_and_utc(self._window.t_rise)}")
        sidecar_path.write_text("\n".join(lines) + "\n")

    def _emit_log_line(self, message: str) -> None:
        # No shared "log" widget on this panel to write into (unlike
        # MountWorker's log events, which app.py routes to the main log
        # box) -- surface via the same stats label used for dropped-frame
        # warnings so a skymap-save failure isn't silently swallowed.
        self._stats_var.set(message)
        self._stats_label.configure(foreground=PALETTE.accent_warn)

    def _on_bit_depth_selected(self, _event: tk.Event) -> None:
        # One setting for both the live/recorded video AND the FITS
        # snapshot -- save_fits_snapshot() just reads whatever depth the
        # stream is currently running at, no separate per-action choice.
        self._camera_worker.set_bit_depth(int(self._bit_depth_var.get()))

    def _set_camera_controls_enabled(self, connected: bool) -> None:
        for widget in self._camera_interactive_widgets:
            widget.configure(state="normal" if connected else "disabled")
        self._bit_depth_combo.configure(state="readonly" if connected else "disabled")

    def _set_camera_roi_bitdepth_enabled(self, enabled: bool) -> None:
        # Only touches ROI/bit-depth widgets, not the whole
        # _camera_interactive_widgets set -- exposure/gain/record/snapshot
        # stay usable while recording, only ROI and bit depth are refused
        # by the worker mid-recording.
        for widget in self._roi_bitdepth_widgets:
            if widget is self._bit_depth_combo:
                widget.configure(state="readonly" if enabled else "disabled")
            else:
                widget.configure(state="normal" if enabled else "disabled")

    def focus_preview(self) -> None:
        """Called by app.py when this tab becomes the active one, so arrow
        keys work immediately without requiring a click first."""
        self._preview_canvas.focus_set()

    def handle_camera_event(self, event: CameraEvent) -> None:
        if event.kind == "connected":
            self._sensor_width = event.payload["width"]
            self._sensor_height = event.payload["height"]
            self._camera_status_var.set(f"Connected — {self._sensor_width}x{self._sensor_height}"
                                         f"{' colour' if event.payload['is_color'] else ' mono'}"
                                         f", {event.payload.get('bit_depth', 8)}-bit capture")
            self._bit_depth_var.set(str(event.payload.get("bit_depth", 8)))
            self._colour_id = event.payload.get("colour_id", 0)
            self._is_color = event.payload.get("is_color", False)
            self._set_camera_controls_enabled(True)
            self._apply_roi(0, 0, self._sensor_width, self._sensor_height)
            self._configure_control_bounds(event.payload.get("controls", {}))
        elif event.kind == "bit_depth_changed":
            self._bit_depth_var.set(str(event.payload["bit_depth"]))
        elif event.kind == "connect_error":
            self._camera_status_var.set(f"Connection failed: {event.payload['message']}")
        elif event.kind == "disconnected":
            self._camera_status_var.set("Not connected (connect in the Connection tab)")
            self._set_camera_controls_enabled(False)
            self._recording = False
            self._record_button.configure(text="Start recording (SER)")
            self._buffer_var.set(0.0)
            self._buffer_pct_var.set("")
            self._file_size_var.set("")
        elif event.kind == "preview_frame":
            self._show_preview_frame(event.payload["pgm"], event.payload["width"], event.payload["height"])
            capacity = event.payload.get("buffer_capacity", 0)
            used = event.payload.get("buffer_used", 0)
            pct = (used / capacity * 100.0) if capacity else 0.0
            self._buffer_var.set(pct)
            self._buffer_pct_var.set(f"{used}/{capacity}" if capacity else "idle")
        elif event.kind == "stats":
            rec = " [RECORDING]" if event.payload["recording"] else ""
            dropped = event.payload.get("dropped_frames", 0)
            errors = event.payload.get("read_errors", 0)
            buffer_dropped = event.payload.get("buffer_dropped_frames", 0)
            self._stats_var.set(
                f"fps: {event.payload['fps']:.1f}   frames recorded: {event.payload['frames_recorded']}"
                f"   dropped: {dropped}   read errors: {errors}   buffer dropped: {buffer_dropped}{rec}"
            )
            # Dropped frames / read errors / buffer drops mean the host
            # isn't keeping up with the sensor (USB bandwidth, exposure/fps
            # mismatch, an actual comm dropout, or -- for buffer drops
            # specifically -- a disk too slow for the write-behind queue
            # to absorb) -- flag it, don't just log it quietly.
            self._stats_label.configure(
                foreground=PALETTE.accent_warn if (dropped or errors or buffer_dropped) else PALETTE.fg,
            )
            self._file_size_var.set(f"file: {format_bytes(event.payload['file_bytes'])}" if event.payload.get("recording") else "")
        elif event.kind == "recording_started":
            self._recording = True
            self._record_button.configure(text="Stop recording")
            self._set_camera_roi_bitdepth_enabled(False)
            self._path_var.set(f"Recording to: {event.payload['path']}")
        elif event.kind == "recording_stopped":
            self._recording = False
            self._record_button.configure(text="Start recording (SER)")
            self._file_size_var.set("")
            self._set_camera_roi_bitdepth_enabled(True)
            buffer_dropped = event.payload.get("buffer_dropped_frames", 0)
            note = f", {buffer_dropped} buffer-dropped" if buffer_dropped else ""
            error = event.payload.get("error")
            if error:
                self._path_var.set(f"Recording stopped early (write error): {error} -- {event.payload['frame_count']} frames saved to {event.payload['path']}")
                self._path_label.configure(foreground=PALETTE.accent_warn)
            else:
                self._path_var.set(f"Saved: {event.payload['path']} ({event.payload['frame_count']} frames{note})")
                self._path_label.configure(foreground=PALETTE.accent_ok)
        elif event.kind == "fits_saved":
            self._path_var.set(f"Saved ({event.payload.get('bit_depth', 8)}-bit): {event.payload['path']}")

    def _configure_control_bounds(self, controls: dict) -> None:
        # Only the widget-level from_/to (not a Variable) needs setting on
        # both scales independently -- the displayed value/label updates
        # via CameraControlVars' shared vars and traces either way.
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

    def _show_preview_frame(self, pgm: bytes, full_width: int, full_height: int) -> None:
        image = fit_pgm_to_canvas(pgm, full_width, full_height, self._preview_canvas)
        self._preview_image = image  # keep a reference — Tk drops images with none
        self._display_w, self._display_h = image.width(), image.height()
        self._display_scale = full_width / self._display_w  # sensor px per displayed px, for ROI drag mapping
        if self._preview_canvas_image_id is None:
            self._preview_canvas_image_id = self._preview_canvas.create_image(0, 0, anchor="nw", image=self._preview_image)
        else:
            self._preview_canvas.itemconfigure(self._preview_canvas_image_id, image=self._preview_image)


class ExposurePanel(ttk.Frame):
    """Rough exposure/gain starting point from the operator's optical train
    + the pass selected in the Passes tab -- see am5/optics.py's module
    docstring for how approximate this is. Pure calculation, no device
    access, so it needs no worker."""

    def __init__(self, parent: tk.Misc):
        super().__init__(parent, padding=10)
        self._trajectory: Trajectory | None = None
        self._window: PassWindow | None = None
        self._preview_image: tk.PhotoImage | None = None  # keep a reference -- Tk drops images with none

        columns = ttk.Frame(self)
        columns.pack(fill="both", expand=True)
        left = ttk.Frame(columns)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(columns)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        form = ttk.Frame(left)
        form.pack(fill="x")
        self._aperture_var = tk.StringVar(value="200")
        self._focal_var = tk.StringVar(value="1000")
        self._barlow_var = tk.StringVar(value="1.0")
        self._pixel_var = tk.StringVar(value="2.9")
        self._trail_var = tk.StringVar(value="1.0")
        fields = [
            ("Aperture (mm)", self._aperture_var), ("Focal length (mm)", self._focal_var),
            ("Barlow multiplier", self._barlow_var), ("Pixel size (um)", self._pixel_var),
            ("Max trail (px)", self._trail_var),
        ]
        for i, (label, var) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w")
            ttk.Entry(form, textvariable=var, width=10).grid(row=i, column=1, sticky="w")

        ttk.Button(left, text="Compute", command=self._on_compute_click).pack(anchor="w", pady=(8, 4))
        self._result_var = tk.StringVar(value="Select a pass in the Passes tab, then Compute.")
        ttk.Label(left, textvariable=self._result_var, justify="left").pack(anchor="w")
        ttk.Label(
            left,
            text="Rough starting point (order-of-magnitude photon budget, no phase-angle/illumination model) --\n"
                 "check the live preview histogram and adjust, don't trust these numbers blindly.",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(anchor="w", pady=(8, 0))

        ttk.Label(right, text="Simulated view at closest approach (real NASA reference photo, resampled to your setup's resolution)").pack(anchor="w")
        self._preview_label = tk.Label(right, background="black")
        self._preview_label.pack(anchor="w", pady=(4, 0))
        self._preview_caption_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self._preview_caption_var, justify="left").pack(anchor="w", pady=(4, 0))

    def set_pass(self, trajectory: Trajectory, window: PassWindow) -> None:
        self._trajectory = trajectory
        self._window = window

    def get_optical_train(self) -> OpticalTrain | None:
        """Whatever's currently typed in this tab's fields, for other
        panels that want the real configured plate scale (e.g. the mock
        camera's simulated star field/ISS size, see TransitPanel's camera
        connect handler) -- None if the fields don't currently parse."""
        try:
            return OpticalTrain(
                aperture_mm=float(self._aperture_var.get()), focal_length_mm=float(self._focal_var.get()),
                barlow_multiplier=float(self._barlow_var.get()), pixel_size_um=float(self._pixel_var.get()),
            )
        except ValueError:
            return None

    def _on_compute_click(self) -> None:
        if self._trajectory is None or self._window is None:
            self._result_var.set("Select a pass in the Passes tab first.")
            return
        try:
            train = OpticalTrain(
                aperture_mm=float(self._aperture_var.get()), focal_length_mm=float(self._focal_var.get()),
                barlow_multiplier=float(self._barlow_var.get()), pixel_size_um=float(self._pixel_var.get()),
            )
            max_trail_px = float(self._trail_var.get())
        except ValueError:
            self._result_var.set("Invalid optical train value(s).")
            return

        peak_speed = float(self._trajectory.sky_speed_deg_s().max())
        exposure_s = max_exposure_s(train, peak_speed, max_trail_px)
        exposure_us = min(exposure_s * 1e6, 1e9)
        distance_km = self._trajectory.distance_at(self._window.t_culminate.timestamp())
        magnitude = estimate_iss_magnitude(distance_km)
        signal_e = estimate_signal_electrons(train, magnitude, exposure_s)
        gain = suggest_gain(signal_e)
        saturation_pct = signal_e / DEFAULT_FULL_WELL_ELECTRONS * 100.0
        gain_note = ""
        if gain == 0.0 and saturation_pct > 100.0:
            gain_note = "  (oversaturated even at gain 0 -- shorten exposure and/or stop down, not enough headroom to add gain)"
        elif gain == 0.0:
            gain_note = "  (already at/near target brightness without extra gain)"

        self._result_var.set(
            f"Plate scale: {train.plate_scale_arcsec_per_px:.2f} arcsec/px\n"
            f"Peak angular speed this pass: {peak_speed:.3f} deg/s\n"
            f"Distance at culmination: {distance_km:.0f} km\n"
            f"Estimated ISS magnitude: {magnitude:+.1f}\n"
            f"Max exposure (motion-limited): {exposure_us:.0f} us\n"
            f"Estimated signal at gain 0: {saturation_pct:.0f}% of full well\n"
            f"Suggested starting gain: {gain:.0f} (0-570){gain_note}"
        )

        closest_km = float(self._trajectory.distance_km.min())
        preview = render_iss_photo(train, closest_km)
        pgm = frame_to_pgm(preview.image)
        self._preview_image = tk.PhotoImage(data=pgm)
        self._preview_label.configure(image=self._preview_image)
        truncation_note = " (too large to fit -- shown cropped)" if preview.truncated else ""
        self._preview_caption_var.set(
            f"Closest approach: {closest_km:.0f} km  --  ISS solar array span: "
            f"{preview.angular_size_arcsec:.1f} arcsec = {preview.camera_px_span:.1f} camera px{truncation_note}"
        )


def _normalize_to_8bit_for_preview(frame: np.ndarray) -> np.ndarray:
    """frame_to_pgm hardcodes an 8-bit (maxval 255) PGM -- a 16-bit SER
    frame needs scaling down first. Auto-stretched per frame (scaled by
    that frame's own max, not a fixed 4095/12-bit assumption) since SER's
    PixelDepth field can in principle be any value up to 16, not
    necessarily always our own camera's 12-bit ADC range -- good enough
    for a quick-look player, not a photometric tool."""
    if frame.dtype == np.uint8:
        return frame
    max_val = int(frame.max())
    if max_val <= 0:
        return np.zeros_like(frame, dtype=np.uint8)
    return (frame.astype(np.float32) * (255.0 / max_val)).astype(np.uint8)


class SerPlayerPanel(ttk.Frame):
    """Standalone SER file viewer -- open any .ser recording (from this
    app's own camera tab or another tool) and scrub/play through its
    frames. Pure local file I/O, no worker/device involved, so unlike
    every other panel it needs no wiring into App._pump_events."""

    def __init__(self, parent: tk.Misc):
        super().__init__(parent, padding=10)
        self._reader: SerReader | None = None
        self._frame_index = 0
        self._playing = False
        self._play_after_id: str | None = None
        self._preview_image: tk.PhotoImage | None = None  # keep a reference -- Tk drops images with none
        self._preview_canvas_image_id: int | None = None

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Button(top, text="Open SER file...", command=self._on_open_click).pack(side="left")
        self._path_var = tk.StringVar(value="No file open")
        ttk.Label(top, textvariable=self._path_var, foreground=PALETTE.fg_dim).pack(side="left", padx=(8, 0))

        self._info_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._info_var, justify="left").pack(anchor="w", pady=(6, 0))

        preview_frame = ttk.LabelFrame(self, text="Frame preview (raw sensor, not debayered)", padding=8)
        preview_frame.pack(fill="both", expand=True, pady=(8, 0))
        self._preview_canvas = tk.Canvas(preview_frame, bg="black", highlightthickness=0)
        self._preview_canvas.pack(fill="both", expand=True)

        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(8, 0))
        self._play_button = ttk.Button(controls, text="Play", command=self._on_play_toggle, state="disabled")
        self._play_button.pack(side="left")
        self._prev_button = ttk.Button(controls, text="< Frame", command=lambda: self._step(-1), state="disabled")
        self._prev_button.pack(side="left", padx=(4, 0))
        self._next_button = ttk.Button(controls, text="Frame >", command=lambda: self._step(1), state="disabled")
        self._next_button.pack(side="left", padx=(4, 0))
        ttk.Label(controls, text="playback fps:").pack(side="left", padx=(10, 0))
        self._fps_var = tk.StringVar(value="10")
        self._fps_entry = ttk.Entry(controls, textvariable=self._fps_var, width=5, state="disabled")
        self._fps_entry.pack(side="left", padx=(4, 0))

        self._frame_var = tk.IntVar(value=0)
        self._frame_scale = ttk.Scale(
            self, from_=0, to=0, variable=self._frame_var, command=self._on_scale_moved, state="disabled",
        )
        self._frame_scale.pack(fill="x", pady=(6, 0))
        self._frame_label_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._frame_label_var).pack(anchor="w")

    def _on_open_click(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Open SER file", filetypes=[("SER video", "*.ser"), ("All files", "*.*")],
        )
        if not path_str:
            return
        self._open_file(Path(path_str))

    def _open_file(self, path: Path) -> None:
        self._stop_playback()
        try:
            reader = SerReader(path)
        except Exception as exc:  # noqa: BLE001 - surface any parse/IO failure to the user, not a crash
            messagebox.showerror("Open SER file", f"Could not open {path.name}:\n{exc}")
            return
        if self._reader is not None:
            self._reader.close()
        self._reader = reader
        self._path_var.set(str(path))

        h = reader.header
        duration_note = ""
        if reader.timestamps and len(reader.timestamps) > 1:
            duration_s = (reader.timestamps[-1] - reader.timestamps[0]).total_seconds()
            avg_fps = (len(reader.timestamps) - 1) / duration_s if duration_s > 0 else 0.0
            duration_note = f"   duration: {duration_s:.1f}s   avg fps: {avg_fps:.1f}"
        elif reader.frame_count > 0:
            duration_note = "   (no timestamp trailer -- interrupted/truncated recording?)"
        self._info_var.set(
            f"{h.width}x{h.height}   {h.colour_name}   {h.pixel_depth}-bit   {h.frame_count} frames\n"
            f"observer: {h.observer or '-'}   instrument: {h.instrument or '-'}   telescope: {h.telescope or '-'}\n"
            f"recorded: {h.date_time_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC{duration_note}"
        )

        has_frames = h.frame_count > 0
        self._frame_scale.configure(to=max(0, h.frame_count - 1), state="normal" if h.frame_count > 1 else "disabled")
        for widget in (self._play_button, self._prev_button, self._next_button, self._fps_entry):
            widget.configure(state="normal" if has_frames else "disabled")
        self._frame_index = 0
        self._frame_var.set(0)
        if has_frames:
            self._show_frame(0)

    def _show_frame(self, index: int) -> None:
        if self._reader is None or self._reader.frame_count == 0:
            return
        index = max(0, min(index, self._reader.frame_count - 1))
        self._frame_index = index
        frame = self._reader.read_frame(index)
        pgm = frame_to_pgm(_normalize_to_8bit_for_preview(frame))
        self._preview_image = fit_pgm_to_canvas(pgm, frame.shape[1], frame.shape[0], self._preview_canvas)
        if self._preview_canvas_image_id is None:
            self._preview_canvas_image_id = self._preview_canvas.create_image(0, 0, anchor="nw", image=self._preview_image)
        else:
            self._preview_canvas.itemconfigure(self._preview_canvas_image_id, image=self._preview_image)

        ts_note = ""
        if self._reader.timestamps is not None:
            ts_note = f"   {self._reader.timestamps[index].strftime('%H:%M:%S.%f')[:-3]} UTC"
        self._frame_label_var.set(f"frame {index + 1}/{self._reader.frame_count}{ts_note}")

    def _on_scale_moved(self, _value: str) -> None:
        if self._playing:
            return  # scale is driven programmatically during playback -- ignore the resulting feedback event
        self._show_frame(int(self._frame_var.get()))

    def _step(self, delta: int) -> None:
        self._stop_playback()
        self._show_frame(self._frame_index + delta)
        self._frame_var.set(self._frame_index)

    def _on_play_toggle(self) -> None:
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        if self._reader is None or self._reader.frame_count <= 1:
            return
        if self._frame_index >= self._reader.frame_count - 1:
            self._frame_index = 0  # replay from the start if already at the end
        self._playing = True
        self._play_button.configure(text="Pause")
        self._play_tick()

    def _stop_playback(self) -> None:
        self._playing = False
        self._play_button.configure(text="Play")
        if self._play_after_id is not None:
            self.after_cancel(self._play_after_id)
            self._play_after_id = None

    def _play_tick(self) -> None:
        if not self._playing or self._reader is None:
            return
        next_index = self._frame_index + 1
        if next_index >= self._reader.frame_count:
            self._stop_playback()
            return
        self._show_frame(next_index)
        self._frame_var.set(next_index)
        try:
            fps = max(0.1, float(self._fps_var.get()))
        except ValueError:
            fps = 10.0
        self._play_after_id = self.after(max(10, round(1000 / fps)), self._play_tick)


class CalibrationPanel(ttk.Frame):
    """Everything calibration-related in one tab, mount-side and camera-side:
    axis-direction calibration (:Me#/:Mn# vs. actual RA/DEC sense), empirical
    mount response-lag measurement, system clock sync status, and the
    camera-based closed-loop guiding calibration. The live auto-guide
    correction itself still runs from here regardless of which tab is
    visible (it needs to keep reading camera frames during an active pass,
    driven by TransitPanel's checkbox) -- only the one-time-per-session
    calibration steps moved into this tab.

    Camera calibration: detects the ISS's blob in the live preview and feeds
    a sky-plane correction into the SAME LiveOffsets a human operator drives
    by hand (arrow keys / delta_t buttons in the Transit tab) -- see
    camera/guiding.py's module docstring. Needs a one-time-per-session
    calibration first: the mapping from "pixels moved on screen" to "arcsec
    moved on sky" depends on how the camera happens to be mounted
    (rotation/mirroring), which isn't knowable in advance."""

    def __init__(
        self, parent: tk.Misc, mount_worker: MountWorker, camera_worker: CameraWorker, live_offsets: LiveOffsets,
        auto_guide_var: tk.BooleanVar | None = None, on_calibration_ready: Callable[[], None] | None = None,
        mount_lag_var: tk.DoubleVar | None = None, axis_signs: AxisSigns | None = None,
    ):
        super().__init__(parent, padding=10)
        self._mount_worker = mount_worker
        self._camera_worker = camera_worker
        self._live_offsets = live_offsets
        # Shared with TransitPanel/JogWindow (same instance, owned by App)
        # when passed -- App auto-corrects dec's sign in place on a pier
        # flip (see AxisSigns.update_pier_side), and the status label below
        # needs to reflect that even when the flip wasn't caused by a
        # fresh Calibrate click. Falls back to a private instance so this
        # panel still works standalone (tests).
        self._axis_signs = axis_signs if axis_signs is not None else AxisSigns(ra=1.0, dec=1.0)
        # Shared with TransitPanel (same instance, owned by App) when passed
        # -- the "Enable auto-guiding" checkbox lives in the Transit tab
        # (it's only useful during an active pass), but this panel is what
        # actually knows whether a blob is in frame and applies the
        # correction, so it still needs to read the same var. Falls back to
        # a private one so this panel still works standalone (tests).
        self._auto_guide_var = auto_guide_var if auto_guide_var is not None else tk.BooleanVar(value=False)
        # Shared with TransitPanel (same instance, owned by App) when passed
        # -- a measured mount_lag_s here is what TrackingConfig.mount_lag_s
        # picks up on the next Start tracking/Simulate click. Falls back to
        # a private var so this panel still works standalone (tests).
        self._mount_lag_var = mount_lag_var if mount_lag_var is not None else tk.DoubleVar(value=0.0)
        # Called once calibration succeeds, so TransitPanel can enable its
        # checkbox -- calibration finishes synchronously inside this panel
        # (button click -> two self.after() timers), not via a worker event,
        # so there's nothing on a queue for App to poll instead.
        self._on_calibration_ready = on_calibration_ready
        self._trajectory: Trajectory | None = None

        self._calibration: GuidingCalibration | None = None
        self._calib_step: str | None = None  # None when idle
        self._calib_ra0: float | None = None
        self._calib_dec0: float | None = None
        self._calib_blob0: BlobDetection | None = None
        self._calib_ra_result: tuple[float, float, float] | None = None  # (d_ra_arcsec, dx_px, dy_px)

        self._latest_radec: tuple[float, float] | None = None
        self._latest_blob: BlobDetection | None = None
        self._preview_image: tk.PhotoImage | None = None
        self._last_correction_t = 0.0

        # Background clock-sync check results land here (subprocess calls,
        # kept off the Tk thread) -- same pattern as ConnectionPanel's city
        # geocoding.
        self._clock_sync_results: "queue.Queue[ClockSyncStatus]" = queue.Queue()

        # Buttons whose command is blocked_while_parked on the worker side
        # (axis calibrate, mount-lag measure, camera-to-sky calibrate --
        # all jog-based). Greyed out while parked/disconnected so a click
        # can never be silently swallowed by that guard -- a real incident:
        # clicking "Measure mount lag" while parked left the button
        # disabled forever, since _handle_measure_mount_lag returns early
        # (just logs a warning) without ever emitting the mount_lag_result
        # event the click handler was waiting for to re-enable it.
        self._motion_widgets: list[tk.Widget] = []
        self._connected = False
        self._parked = False

        self._build_mount_calibration_section(self)

        columns = ttk.Frame(self)
        columns.pack(fill="both", expand=True)
        left = ttk.Frame(columns)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(columns)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        ttk.Label(
            left,
            text="Camera-based auto-guiding: detects the ISS in the live preview and\n"
                 "nudges the mount to keep it centered, on top of the usual feedforward tracking.",
            justify="left",
        ).pack(anchor="w")

        ttk.Label(
            left,
            text="1) Point the camera at any bright, steady object (a star with sidereal\n"
                 "   tracking on, or even a distant light at night) -- it does NOT need to\n"
                 "   be the ISS, this only measures how the camera is physically mounted.\n"
                 "2) Click Calibrate. The mount nudges each axis briefly and measures how\n"
                 "   far the blob moves on screen -- do this once per session (as long as\n"
                 "   the camera's orientation/focal train doesn't change, it stays valid).",
            justify="left", foreground=PALETTE.fg_dim,
        ).pack(anchor="w", pady=(4, 8))

        speed_row = ttk.Frame(left)
        speed_row.pack(anchor="w", pady=(0, 8))
        ttk.Label(speed_row, text="Nudge rate (x sidereal)").pack(side="left")
        self._calib_rate_var = tk.StringVar(value=str(GUIDING_CALIB_NUDGE_RATE_X))
        ttk.Entry(speed_row, textvariable=self._calib_rate_var, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(speed_row, text="Nudge duration (s)").pack(side="left")
        self._calib_duration_var = tk.StringVar(value=str(GUIDING_CALIB_NUDGE_DURATION_S))
        ttk.Entry(speed_row, textvariable=self._calib_duration_var, width=6).pack(side="left", padx=(4, 0))
        ttk.Label(
            left,
            text="Lower these for a narrow field of view (long focal length) -- the\n"
                 "defaults can push the target out of frame before it's measured.",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(anchor="w", pady=(0, 8))

        self._calibrate_button = ttk.Button(left, text="Calibrate camera-to-sky mapping", command=self._on_calibrate_click)
        self._calibrate_button.pack(anchor="w")
        self._motion_widgets.append(self._calibrate_button)
        self._calib_status_var = tk.StringVar(value="Not calibrated this session")
        ttk.Label(left, textvariable=self._calib_status_var).pack(anchor="w", pady=(2, 8))

        ttk.Label(
            left,
            text="Once calibrated, enable auto-guiding from the Transit tab (it's only\n"
                 "useful during an active pass). Only corrects cross-track (sideways drift)\n"
                 "automatically -- along-track (ahead/behind) stays a manual delta_t call,\n"
                 "same reasoning as the Transit tab (a single image can't tell a clock\n"
                 "offset from mount rate-change lag).",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(anchor="w", pady=(4, 0))

        self._blob_status_var = tk.StringVar(value="No frame yet")
        ttk.Label(left, textvariable=self._blob_status_var, justify="left").pack(anchor="w", pady=(8, 0))

        ttk.Label(right, text="Live preview (crosshair = detected ISS position)").pack(anchor="w")
        self._preview_label = tk.Label(right, background="black")
        self._preview_label.pack(anchor="w", pady=(4, 0))

        self.set_connected(False)
        self.after(200, self._poll_clock_sync_results)

    # -- mount calibration section (axis signs, mount lag, clock sync) -------

    def _build_mount_calibration_section(self, parent: tk.Misc) -> None:
        frame = ttk.LabelFrame(parent, text="Mount calibration", padding=8)
        frame.pack(anchor="w", fill="x", pady=(0, 10))
        columns = ttk.Frame(frame)
        columns.pack(fill="x")
        axis_col = ttk.Frame(columns)
        axis_col.pack(side="left", anchor="n", padx=(0, 24))
        lag_col = ttk.Frame(columns)
        lag_col.pack(side="left", anchor="n", padx=(0, 24))
        clock_col = ttk.Frame(columns)
        clock_col.pack(side="left", anchor="n")

        ttk.Label(axis_col, text="Axis directions", font=("", 9, "bold")).pack(anchor="w")
        ttk.Label(
            axis_col, text="Jogs each axis briefly and checks whether\n"
                           "reported RA/DEC moved the expected way. Works\n"
                           "from either pier side (E/W) -- no need to move\n"
                           "the mount first, a pier flip later auto-adjusts\n"
                           "the DEC direction on its own.",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(anchor="w", pady=(0, 4))
        self._axis_calibrate_button = ttk.Button(axis_col, text="Calibrate axis directions", command=self._mount_worker.calibrate)
        self._axis_calibrate_button.pack(anchor="w")
        self._motion_widgets.append(self._axis_calibrate_button)
        self._axis_calibration_var = tk.StringVar(value="Not calibrated this session")
        ttk.Label(axis_col, textvariable=self._axis_calibration_var).pack(anchor="w", pady=(2, 0))

        ttk.Label(lag_col, text="Mount response lag", font=("", 9, "bold")).pack(anchor="w")
        ttk.Label(
            lag_col, text="Commands a step rate on RA and times how long\n"
                          "it takes to reach steady speed -- feed the result\n"
                          "into mount_lag_s (below) to feedforward-lead the\n"
                          "commanded rate by that much.",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(anchor="w", pady=(0, 4))
        lag_params_row = ttk.Frame(lag_col)
        lag_params_row.pack(anchor="w", pady=(0, 4))
        ttk.Label(lag_params_row, text="rate (x sidereal)").pack(side="left")
        self._lag_rate_var = tk.StringVar(value="100")
        ttk.Entry(lag_params_row, textvariable=self._lag_rate_var, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(lag_params_row, text="duration (s)").pack(side="left")
        self._lag_duration_var = tk.StringVar(value="1.5")
        ttk.Entry(lag_params_row, textvariable=self._lag_duration_var, width=6).pack(side="left", padx=(4, 0))
        self._lag_measure_button = ttk.Button(lag_col, text="Measure mount lag", command=self._on_measure_lag_click)
        self._lag_measure_button.pack(anchor="w")
        self._motion_widgets.append(self._lag_measure_button)
        self._lag_status_var = tk.StringVar(value="Not measured this session")
        ttk.Label(lag_col, textvariable=self._lag_status_var, justify="left").pack(anchor="w", pady=(2, 4))
        mount_lag_row = ttk.Frame(lag_col)
        mount_lag_row.pack(anchor="w")
        ttk.Label(mount_lag_row, text="mount_lag_s used by tracking:").pack(side="left")
        ttk.Entry(mount_lag_row, textvariable=self._mount_lag_var, width=6).pack(side="left", padx=(4, 0))

        ttk.Label(clock_col, text="System clock sync", font=("", 9, "bold")).pack(anchor="w")
        ttk.Label(
            clock_col, text="A clock offset produces the same symptom as a\n"
                            "real mount lag (a stable along-track error) --\n"
                            "cheap to rule out first.",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(anchor="w", pady=(0, 4))
        self._clock_sync_button = ttk.Button(clock_col, text="Check clock sync", command=self._on_check_clock_sync_click)
        self._clock_sync_button.pack(anchor="w")
        self._clock_sync_var = tk.StringVar(value="Not checked this session")
        ttk.Label(clock_col, textvariable=self._clock_sync_var, justify="left", wraplength=220).pack(anchor="w", pady=(2, 0))

    def _on_measure_lag_click(self) -> None:
        try:
            rate_x = float(self._lag_rate_var.get())
            duration_s = float(self._lag_duration_var.get())
        except ValueError:
            self._lag_status_var.set("Invalid rate/duration")
            return
        self._lag_measure_button.configure(state="disabled")
        self._lag_status_var.set("Measuring (jogs RA briefly)...")
        self._mount_worker.measure_mount_lag(rate_x=rate_x, duration_s=duration_s)

    def _on_check_clock_sync_click(self) -> None:
        self._clock_sync_button.configure(state="disabled")
        self._clock_sync_var.set("Checking...")
        threading.Thread(target=self._check_clock_sync_bg, daemon=True).start()

    def _check_clock_sync_bg(self) -> None:
        # subprocess.run calls (up to ~3 tools tried in sequence) -- off the
        # Tk thread, result marshaled back via the queue below, same
        # reasoning as ConnectionPanel's background geocoding.
        self._clock_sync_results.put(check_clock_sync())

    def _poll_clock_sync_results(self) -> None:
        try:
            status = self._clock_sync_results.get_nowait()
        except queue.Empty:
            pass
        else:
            self._clock_sync_button.configure(state="normal")
            if status.synchronized is True:
                offset = f" (offset {status.offset_s * 1000:+.1f} ms)" if status.offset_s is not None else ""
                self._clock_sync_var.set(f"Synchronized{offset} -- via {status.source}")
            elif status.synchronized is False:
                offset = f" (offset {status.offset_s:+.2f} s)" if status.offset_s is not None else ""
                self._clock_sync_var.set(f"NOT synchronized{offset} -- via {status.source}")
            else:
                self._clock_sync_var.set(f"Unknown -- {status.detail}")
        self.after(200, self._poll_clock_sync_results)

    # -- wiring from app.py --------------------------------------------------

    def set_trajectory(self, trajectory: Trajectory) -> None:
        self._trajectory = trajectory

    def _refresh_widget_states(self) -> None:
        state = "normal" if (self._connected and not self._parked) else "disabled"
        for widget in self._motion_widgets:
            widget.configure(state=state)

    def set_connected(self, connected: bool) -> None:
        self._connected = connected
        if not connected:
            self._parked = False
        self._refresh_widget_states()

    # -- event handlers -------------------------------------------------------

    def _render_axis_calibration_status(self, ra_sign: float, dec_sign: float, pier_side: str | None) -> None:
        side_text = f" (pier side {pier_side})" if pier_side else ""
        self._axis_calibration_var.set(f"RA sign: {ra_sign:+.0f}  DEC sign: {dec_sign:+.0f}{side_text}")

    def handle_mount_event(self, event: WorkerEvent) -> None:
        if event.kind == "position":
            self._latest_radec = (event.payload["ra_hours"], event.payload["dec_deg"])
            # Reflects an App-level pier-flip auto-correction (see
            # AxisSigns.update_pier_side) even when it wasn't triggered by
            # a fresh Calibrate click here -- only once a calibration has
            # actually happened this session, so this never overwrites
            # "Not calibrated this session" with a misleading default.
            if self._axis_signs.calibrated_pier_side is not None:
                self._render_axis_calibration_status(
                    self._axis_signs.ra, self._axis_signs.dec, self._axis_signs.calibrated_pier_side,
                )
        elif event.kind == "tracking_tick":
            actual_ra_deg = event.payload["actual_ra_deg"]
            if actual_ra_deg != "":  # only populated every error_log_every ticks, see tracker.py
                self._latest_radec = (actual_ra_deg / 15.0, event.payload["actual_dec_deg"])
        elif event.kind == "calibration_done":
            # Mutate self._axis_signs directly (mirrors TransitPanel.set_
            # axis_signs) rather than only rendering from the event payload
            # -- keeps this panel self-consistent whether or not it's
            # sharing App's instance, and means the "position" branch
            # above can always trust self._axis_signs is current.
            self._axis_signs.ra = event.payload["ra_sign"]
            self._axis_signs.dec = event.payload["dec_sign"]
            self._axis_signs.calibrated_pier_side = event.payload.get("pier_side")
            self._render_axis_calibration_status(
                self._axis_signs.ra, self._axis_signs.dec, self._axis_signs.calibrated_pier_side,
            )
        elif event.kind == "mount_lag_result":
            lag_s = event.payload["lag_s"]
            self._lag_status_var.set(
                f"lag {lag_s:.3f}s, steady rate {event.payload['steady_rate_arcsec_s']:+.1f}\"/s "
                f"({event.payload['samples']} samples) -- mount_lag_s updated below"
            )
            self._mount_lag_var.set(round(lag_s, 3))
            self._refresh_widget_states()  # not a flat "normal" -- respects a park that landed mid-measurement
        elif event.kind == "parked":
            self._parked = True
            self._refresh_widget_states()
        elif event.kind == "unparked":
            self._parked = False
            self._refresh_widget_states()

    def handle_camera_event(self, event: CameraEvent) -> None:
        if event.kind != "preview_frame":
            return
        frame = pgm_to_array(event.payload["pgm"])
        # Detect on a downsampled copy, not the full-resolution frame --
        # same fix as FinderState.update_frame (see its own docstring for
        # the original incident): detect_brightest_blob's centroid math
        # costs ~60ms on a finder-class sensor (confirmed measured, vs
        # ~9ms at this project's normal ~2MP main-camera resolution), and
        # building a full-resolution tk.PhotoImage for the preview below
        # adds more on top -- together enough to blow the ~100ms preview
        # interval and freeze the whole Tk main thread once App._pump_
        # events has a backlog of queued preview_frame events to drain.
        # Never a problem at this project's own reference main camera
        # (ASI290MC, ~2MP) -- only surfaced when a wider/higher-res sensor
        # (e.g. an ASI678MM-class camera) is used in the main role instead.
        _dw, _dh, scale, small = downsample_for_display(frame, MAX_CALIBRATION_PREVIEW_DIM, MAX_CALIBRATION_PREVIEW_DIM)
        small_blob = detect_brightest_blob(small)
        # self._latest_blob (and everything downstream: the calibration
        # sequence's pixel-delta math, auto-guide's dx_px/dy_px against
        # frame.shape) stays in FULL-resolution pixel coordinates, exactly
        # as before this fix -- only the detection pass itself runs on the
        # downsampled copy, so nothing downstream needs to know that.
        blob = small_blob
        if small_blob.found:
            blob = dataclasses.replace(small_blob, centroid_x=small_blob.centroid_x / scale, centroid_y=small_blob.centroid_y / scale)
        self._latest_blob = blob
        self._show_preview(small, small_blob)
        if blob.found:
            self._blob_status_var.set(f"ISS at pixel ({blob.centroid_x:.0f}, {blob.centroid_y:.0f}), peak {blob.peak_value:.0f}")
        else:
            self._blob_status_var.set("ISS not detected in frame")
        if self._auto_guide_var.get():
            self._maybe_apply_auto_guide_correction(frame.shape, blob)

    def _show_preview(self, frame: np.ndarray, blob: BlobDetection) -> None:
        # frame is the already-downsampled display copy from
        # handle_camera_event, and blob is in that SAME (downsampled)
        # coordinate space -- do not pass full-resolution coordinates here.
        display = frame.copy()
        h, w = display.shape
        if blob.found:
            cx = int(round(min(max(blob.centroid_x, 0), w - 1)))
            cy = int(round(min(max(blob.centroid_y, 0), h - 1)))
            size = 8
            display[cy, max(0, cx - size) : min(w, cx + size)] = 255
            display[max(0, cy - size) : min(h, cy + size), cx] = 255
        self._preview_image = tk.PhotoImage(data=frame_to_pgm(display))
        self._preview_label.configure(image=self._preview_image)

    # -- calibration sequence --------------------------------------------------

    def _calib_rate_x(self) -> float:
        try:
            return float(self._calib_rate_var.get())
        except ValueError:
            return GUIDING_CALIB_NUDGE_RATE_X

    def _calib_duration_s(self) -> float:
        try:
            return float(self._calib_duration_var.get())
        except ValueError:
            return GUIDING_CALIB_NUDGE_DURATION_S

    def _on_calibrate_click(self) -> None:
        if self._calib_step is not None:
            return
        if self._latest_radec is None or self._latest_blob is None or not self._latest_blob.found:
            self._calib_status_var.set("Can't calibrate: no bright object detected in the current frame yet.")
            return
        self._calib_step = "ra"
        self._calibrate_button.configure(state="disabled")
        self._calib_status_var.set("Calibrating RA axis -- nudging east...")
        self._calib_ra0, self._calib_dec0 = self._latest_radec
        self._calib_blob0 = self._latest_blob
        self._mount_worker.jog_start("e", rate_x=self._calib_rate_x())
        self.after(int(self._calib_duration_s() * 1000), self._calib_ra_stop)

    def _calib_ra_stop(self) -> None:
        self._mount_worker.jog_stop("e")
        self.after(int(GUIDING_CALIB_SETTLE_S * 1000), self._calib_ra_measure)

    def _calib_ra_measure(self) -> None:
        if not self._calib_measurement_ok():
            return
        ra1, dec1 = self._latest_radec
        blob1 = self._latest_blob
        d_ra_arcsec = circular_diff_hours(ra1, self._calib_ra0) * 15.0 * 3600.0 * math.cos(math.radians(dec1))
        dx, dy = blob1.centroid_x - self._calib_blob0.centroid_x, blob1.centroid_y - self._calib_blob0.centroid_y
        if abs(d_ra_arcsec) < 1.0:
            self._calib_status_var.set("Calibration failed: mount didn't move measurably in RA -- check connection.")
            self._calib_step = None
            self._refresh_widget_states()
            return
        self._calib_ra_result = (d_ra_arcsec, dx, dy)

        self._calib_step = "dec"
        self._calib_status_var.set("Calibrating DEC axis -- nudging north...")
        self._calib_ra0, self._calib_dec0 = self._latest_radec
        self._calib_blob0 = self._latest_blob
        self._mount_worker.jog_start("n", rate_x=self._calib_rate_x())
        self.after(int(self._calib_duration_s() * 1000), self._calib_dec_stop)

    def _calib_dec_stop(self) -> None:
        self._mount_worker.jog_stop("n")
        self.after(int(GUIDING_CALIB_SETTLE_S * 1000), self._calib_dec_measure)

    def _calib_dec_measure(self) -> None:
        if not self._calib_measurement_ok():
            return
        ra1, dec1 = self._latest_radec
        blob1 = self._latest_blob
        d_dec_arcsec = (dec1 - self._calib_dec0) * 3600.0
        dx, dy = blob1.centroid_x - self._calib_blob0.centroid_x, blob1.centroid_y - self._calib_blob0.centroid_y
        self._refresh_widget_states()
        self._calib_step = None
        if abs(d_dec_arcsec) < 1.0:
            self._calib_status_var.set("Calibration failed: mount didn't move measurably in DEC -- check connection.")
            return
        d_ra_arcsec, dx1, dy1 = self._calib_ra_result
        try:
            self._calibration = calibrate_from_nudges(d_ra_arcsec, dx1, dy1, d_dec_arcsec, dx, dy)
        except ValueError as exc:
            self._calib_status_var.set(f"Calibration failed: {exc}")
            return
        self._calib_status_var.set(f"Calibrated: ~{self._calibration.arcsec_per_pixel:.2f} arcsec/px -- ready to auto-guide")
        if self._on_calibration_ready is not None:
            self._on_calibration_ready()

    def _calib_measurement_ok(self) -> bool:
        if self._latest_radec is None or self._latest_blob is None or not self._latest_blob.found:
            self._calib_status_var.set("Calibration failed: lost the blob mid-calibration -- retry with a brighter/steadier target.")
            self._calib_step = None
            self._refresh_widget_states()
            return False
        return True

    # -- auto-guide correction --------------------------------------------------

    def _maybe_apply_auto_guide_correction(self, frame_shape: tuple[int, int], blob: BlobDetection) -> None:
        """dx_px/dy_px (blob minus frame center) represents "where the ISS
        actually is" relative to "where the camera boresight (== mount
        pointing) currently is" -- i.e. a (target - actual) sky offset once
        run through the calibration. decompose_error's cross-track sign
        convention already matches trigger_perp_pulse's (see am5/tracker.py
        decompose_error's docstring / the reasoning worked out when this
        panel was built): a positive cross_deg needs a sign=+1 pulse to
        close the gap, no inversion needed."""
        if not blob.found or self._calibration is None or self._trajectory is None:
            return
        if time.monotonic() - self._last_correction_t < GUIDING_MIN_CORRECTION_INTERVAL_S:
            return
        height, width = frame_shape
        dx_px = blob.centroid_x - width / 2.0
        dy_px = blob.centroid_y - height / 2.0
        if math.hypot(dx_px, dy_px) < GUIDING_DEADBAND_PX:
            return
        try:
            d_ra_arcsec, d_dec_arcsec = self._calibration.pixel_to_sky(dx_px, dy_px)
        except ValueError:
            return
        _, dec_deg, dra_dt, ddec_dt = self._trajectory.interpolate(time.time())
        _, cross_deg = decompose_error(d_ra_arcsec / 3600.0, d_dec_arcsec / 3600.0, dec_deg, dra_dt, ddec_dt)
        if abs(cross_deg * 3600.0) < GUIDING_DEADBAND_PX * self._calibration.arcsec_per_pixel:
            return
        self._last_correction_t = time.monotonic()
        self._live_offsets.trigger_perp_pulse(1.0 if cross_deg > 0 else -1.0, duration_s=GUIDING_PERP_PULSE_DURATION_S)


# ---------------------------------------------------------------------------
# FinderCameraPanel
# ---------------------------------------------------------------------------

class FinderCameraPanel(ttk.Frame):
    """Wide-field finder-scope camera for ISS acquisition.

    Completely optional -- if no finder camera is plugged in, this panel
    stays greyed out and nothing else changes.  When a camera IS connected:

    1. Live preview with ISS blob highlight (bright moving dot).
    2. "Calibrate fields" button: point both cameras at the same region,
       click -- FFT cross-correlation stores the offset between them.
    3. Once calibrated, enabling "Finder correction" in the Transit tab
       automatically nudges the mount cross-track whenever the ISS blob
       drifts away from the calibrated boresight offset.
    """

    def __init__(
        self,
        parent: tk.Misc,
        finder_worker: CameraWorker,
        finder_state: FinderState,
        live_offsets: LiveOffsets | None = None,
        on_calibration_ready: Callable[[], None] | None = None,
    ):
        super().__init__(parent, padding=10)
        self._finder_worker = finder_worker
        # Calibration here finishes entirely on its own (button click ->
        # FFT correlation, synchronous) -- same rationale as CalibrationPanel's
        # own on_calibration_ready: there's no worker event to pump this
        # through, so it has to tell TransitPanel directly when its "Enable
        # finder correction" checkbox should become enabled.
        self._on_calibration_ready = on_calibration_ready
        # The main camera's own frames/plate scale arrive via finder_state
        # (last_main_frame, main_plate_scale_arcsec -- see camera/finder.py
        # and App._pump_events, which feeds them from the main CameraWorker's
        # own events) rather than this panel holding a second CameraWorker
        # reference directly -- one less thing for this panel to wire up.
        self._finder_state = finder_state
        # Shared with TransitPanel (same instance, owned by App) when
        # passed -- so a delta_t/perp nudge made from here (see the
        # tracking-offset controls below) lands in the SAME LiveOffsets
        # the active tracking loop is reading, exactly like TransitPanel's
        # own controls. Falls back to a private instance so this panel
        # still works standalone (tests, or a build without the Transit
        # tab wired in) -- same rationale as TransitPanel's own fallback.
        self._live_offsets = live_offsets if live_offsets is not None else LiveOffsets()
        self._connected = False
        self._latest_frame: np.ndarray | None = None
        self._photo: tk.PhotoImage | None = None
        self._interactive: list[tk.Widget] = []

        ttk.Label(
            self,
            text=(
                "Wide-field finder camera -- helps acquire the ISS when the main camera's FOV is too "
                "narrow. Connect it in the Connection tab, then use 'Calibrate fields' below to measure "
                "the offset between the two cameras."
            ),
            foreground=PALETTE.fg_dim, wraplength=800, justify="left",
        ).pack(anchor="w")

        # Connection state is read-only here -- actually connecting happens
        # in ConnectionPanel, alongside the mount and main camera, so all
        # three devices are managed in one consistent place.
        self._status_var = tk.StringVar(value="Not connected -- connect in the Connection tab")
        ttk.Label(self, textvariable=self._status_var, foreground=PALETTE.fg_dim).pack(anchor="w", pady=(4, 0))

        # Tracking offset controls -- same LiveOffsets as TransitPanel's own
        # (see this panel's __init__ docstring comment on self._live_offsets),
        # duplicated here because once a pass starts, the finder's wide field
        # is where the operator is actually watching to get the ISS into the
        # much narrower acquisition camera's FOV -- switching to the Transit
        # tab just to nudge would mean looking away at exactly the moment
        # framing matters most.
        offset_frame = ttk.LabelFrame(self, text="Tracking offset (same as Transit tab)", padding=8)
        offset_frame.pack(fill="x", pady=(8, 0))
        offset_row = ttk.Frame(offset_frame)
        offset_row.pack(anchor="w")
        ttk.Label(offset_row, text="delta_t:").pack(side="left")
        ttk.Button(offset_row, text="-1s", width=4, command=lambda: self._live_offsets.adjust_delta_t(-1.0)).pack(side="left")
        self._finder_delta_t_minus_button = ttk.Button(
            offset_row, text="-0.1s", width=5, command=lambda: self._live_offsets.adjust_delta_t(-0.1),
        )
        self._finder_delta_t_minus_button.pack(side="left")
        self._finder_delta_t_var = tk.StringVar(value="+0.0s")
        ttk.Label(offset_row, textvariable=self._finder_delta_t_var, width=8).pack(side="left")
        self._finder_delta_t_plus_button = ttk.Button(
            offset_row, text="+0.1s", width=5, command=lambda: self._live_offsets.adjust_delta_t(0.1),
        )
        self._finder_delta_t_plus_button.pack(side="left")
        ttk.Button(offset_row, text="+1s", width=4, command=lambda: self._live_offsets.adjust_delta_t(1.0)).pack(side="left")

        perp_row = ttk.Frame(offset_frame)
        perp_row.pack(anchor="w", pady=(4, 0))
        ttk.Label(perp_row, text="perpendicular nudge:").pack(side="left")
        self._finder_perp_left_button = ttk.Button(perp_row, text="<", width=3, command=lambda: self._live_offsets.trigger_perp_pulse(-1.0))
        self._finder_perp_left_button.pack(side="left")
        self._finder_perp_right_button = ttk.Button(perp_row, text=">", width=3, command=lambda: self._live_offsets.trigger_perp_pulse(1.0))
        self._finder_perp_right_button.pack(side="left")
        ttk.Label(offset_frame, text="(↑ ↓ = delta_t, ← → = nudge -- from anywhere in this tab)",
                  foreground=PALETTE.fg_dim).pack(anchor="w", pady=(2, 0))

        # Exposure / gain -- ASI 678MM defaults, log-scale slider like main camera
        import math as _math
        exp_frame = ttk.LabelFrame(self, text="Exposure / gain", padding=8)
        exp_frame.pack(fill="x", pady=(8, 0))
        # log10(50000) ≈ 4.7 -- 50ms default for finder
        self._finder_exp_log = tk.DoubleVar(value=_math.log10(50000))
        self._finder_exp_label = tk.StringVar(value=format_exposure_us(50000))
        self._finder_gain_var = tk.DoubleVar(value=100)
        self._finder_gain_label = tk.StringVar(value="100")

        exp_row = ttk.Frame(exp_frame)
        exp_row.pack(fill="x")
        ttk.Label(exp_row, text="Exp", width=4).pack(side="left")
        self._finder_exp_scale = ttk.Scale(
            exp_row, from_=1.5, to=_math.log10(MAX_FINDER_EXPOSURE_US), variable=self._finder_exp_log, state="disabled",
            command=lambda _v: (
                self._finder_exp_label.set(format_exposure_us(10 ** self._finder_exp_log.get())),
                self._apply_camera_settings() if self._connected else None,
            ),
        )
        self._finder_exp_scale.pack(side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Label(exp_row, textvariable=self._finder_exp_label, width=10).pack(side="left")

        gain_row = ttk.Frame(exp_frame)
        gain_row.pack(fill="x", pady=(4, 0))
        ttk.Label(gain_row, text="Gain", width=4).pack(side="left")
        self._finder_gain_scale = ttk.Scale(
            gain_row, from_=0, to=570, variable=self._finder_gain_var, state="disabled",
            command=lambda _v: (
                self._finder_gain_label.set(str(round(self._finder_gain_var.get()))),
                self._apply_camera_settings() if self._connected else None,
            ),
        )
        self._finder_gain_scale.pack(side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Label(gain_row, textvariable=self._finder_gain_label, width=6).pack(side="left")

        # Calibration
        calib_frame = ttk.LabelFrame(self, text="Field calibration", padding=8)
        calib_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(
            calib_frame,
            text="Point both cameras at the same star field, then click Calibrate.\n"
                 "The FFT cross-correlation takes ~1 s and stores the offset permanently for this session.",
            foreground=PALETTE.fg_dim, justify="left",
        ).pack(anchor="w")
        calib_row = ttk.Frame(calib_frame)
        calib_row.pack(anchor="w", pady=(6, 0))
        # Read-only -- the real plate scales come from whatever optics were
        # actually configured in the Connection tab at connect time (see
        # FinderState.main_plate_scale_arcsec/finder_plate_scale_arcsec),
        # not a separately-typed field here that could silently drift out
        # of sync with the real configuration.
        self._scales_status_var = tk.StringVar(value="")
        ttk.Label(calib_row, textvariable=self._scales_status_var, foreground=PALETTE.fg_dim).pack(side="left", padx=(0, 8))
        # Manual, NOT auto-measured: phase cross-correlation only recovers
        # a translation, and this project's star fields are too sparse (a
        # handful of point sources, not a textured scene) for FFT-based
        # rotation/scale registration to be reliable -- see
        # FinderCalibration.calibrate_from_frames' own docstring. Defaults
        # to 0 (the old, silent assumption) but now visible/editable
        # instead of hidden, so a real mechanical roll offset between the
        # two scopes can actually be corrected for once measured by hand.
        ttk.Label(calib_row, text="Rotation (deg):").pack(side="left")
        self._finder_rotation_var = tk.StringVar(value="0.0")
        ttk.Entry(calib_row, textvariable=self._finder_rotation_var, width=6).pack(side="left", padx=(4, 8))
        self._calib_btn = ttk.Button(calib_row, text="Calibrate fields", command=self._on_calibrate, state="disabled")
        self._calib_btn.pack(side="left")
        self._interactive.append(self._calib_btn)
        self._calib_status_var = tk.StringVar(value="Not calibrated")
        ttk.Label(calib_frame, textvariable=self._calib_status_var, foreground=PALETTE.fg_dim).pack(anchor="w", pady=(4, 0))

        # Live preview
        preview_frame = ttk.LabelFrame(self, text="Finder preview -- ISS blob highlighted in red", padding=8)
        preview_frame.pack(fill="both", expand=True, pady=(8, 0))
        self._canvas = tk.Canvas(preview_frame, bg="black", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._blob_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._blob_var, foreground=PALETTE.accent_ok).pack(anchor="w", pady=(4, 0))
        self._refresh_scales_status()

        # Same rationale as TransitPanel's own _bind_offset_keys call --
        # bound recursively on every widget, not just self, since a
        # binding on a container does NOT fire just because some
        # descendant happens to have focus (Tk only consults the actually
        # focused widget's own bindtags).
        self._bind_offset_keys(self)
        self.after(300, self._poll_delta_t_display)

    # ------------------------------------------------------------------

    def _apply_camera_settings(self) -> None:
        if not self._connected:
            return
        exp_us = round(10 ** self._finder_exp_log.get())
        gain = round(self._finder_gain_var.get())
        self._finder_worker.set_exposure_us(exp_us)
        self._finder_worker.set_gain(gain)

    def _refresh_scales_status(self) -> None:
        self._scales_status_var.set(
            f"finder {self._finder_state.finder_plate_scale_arcsec:.3f}\"/px, "
            f"main {self._finder_state.main_plate_scale_arcsec:.3f}\"/px "
            "(from the Connection tab)"
        )

    def _poll_delta_t_display(self) -> None:
        dt, _ = self._live_offsets.snapshot()
        self._finder_delta_t_var.set(f"{dt:+.1f}s")
        self.after(300, self._poll_delta_t_display)

    def _bind_offset_keys(self, widget: tk.Misc) -> None:
        widget.bind("<Left>", lambda _e: self._on_finder_perp_nudge_key_press(-1.0))
        widget.bind("<Right>", lambda _e: self._on_finder_perp_nudge_key_press(1.0))
        widget.bind("<Up>", lambda _e: self._on_finder_delta_t_key_press(0.1))
        widget.bind("<Down>", lambda _e: self._on_finder_delta_t_key_press(-0.1))
        for child in widget.winfo_children():
            self._bind_offset_keys(child)

    def _on_finder_perp_nudge_key_press(self, sign: float) -> str:
        self._live_offsets.trigger_perp_pulse(sign)
        self._flash_button(self._finder_perp_left_button if sign < 0 else self._finder_perp_right_button)
        return "break"  # pre-empts the focused widget's own Left/Right handling

    def _on_finder_delta_t_key_press(self, step: float) -> str:
        self._live_offsets.adjust_delta_t(step)
        self._flash_button(self._finder_delta_t_plus_button if step > 0 else self._finder_delta_t_minus_button)
        return "break"  # pre-empts the focused widget's own Up/Down handling

    def _flash_button(self, button: ttk.Button, duration_ms: int = int(GUIDING_PERP_PULSE_DURATION_S * 1000)) -> None:
        """Briefly shows a button as pressed -- for actions that fire a
        single short pulse rather than a press-and-hold, so there's
        something for the keyboard-triggered case to visually attach to."""
        button.state(["pressed"])
        self.after(duration_ms, lambda: button.state(["!pressed"]))

    def _on_calibrate(self) -> None:
        """Cross-correlate the latest finder frame against the most recent
        main camera frame to compute the field offset."""
        if self._latest_frame is None:
            self._calib_status_var.set("No finder frame yet -- wait for preview")
            return
        main_frame = self._finder_state.last_main_frame
        used_fallback = main_frame is None
        if used_fallback:
            # Fall back: calibrate finder to itself (offset = 0, still useful
            # for ISS blob → correction when both cameras share the boresight)
            main_frame = self._latest_frame
        try:
            rotation_deg = float(self._finder_rotation_var.get())
        except ValueError:
            self._calib_status_var.set("Invalid rotation value")
            return
        finder_scale = self._finder_state.finder_plate_scale_arcsec
        main_scale = self._finder_state.main_plate_scale_arcsec
        self._calib_status_var.set("Calibrating…")
        self.update_idletasks()
        try:
            self._finder_state.calibration.calibrate_from_frames(
                main_frame, self._latest_frame,
                main_plate_scale_arcsec=main_scale,
                finder_plate_scale_arcsec=finder_scale,
                rotation_deg=rotation_deg,
            )
            dr = self._finder_state.calibration.offset_row
            dc = self._finder_state.calibration.offset_col
            fallback_note = (
                " -- WARNING: no main camera frame yet, calibrated the finder against itself "
                "(offset is meaningless -- point both cameras at the same field and retry)"
                if used_fallback else ""
            )
            self._calib_status_var.set(
                f"Calibrated ✓  offset ({dr:+.1f}, {dc:+.1f}) finder px  "
                f"scale ratio {self._finder_state.calibration.plate_scale_ratio:.2f}  "
                f"rotation {rotation_deg:+.1f}°{fallback_note}"
            )
            if self._on_calibration_ready is not None:
                self._on_calibration_ready()
        except Exception as exc:  # noqa: BLE001
            self._calib_status_var.set(f"Calibration failed: {exc}")

    def handle_camera_event(self, event: CameraEvent) -> None:
        if event.kind == "connected":
            self._connected = True
            w, h = event.payload["width"], event.payload["height"]
            self._status_var.set(f"Connected — {w}×{h} {'colour' if event.payload['is_color'] else 'mono'}")
            self._calib_btn.configure(state="normal")
            self._finder_exp_scale.configure(state="normal")
            self._finder_gain_scale.configure(state="normal")
            self._apply_camera_settings()
            self._refresh_scales_status()
        elif event.kind == "connect_error":
            self._status_var.set(f"Error: {event.payload.get('message', '?')} -- retry in the Connection tab")
        elif event.kind == "disconnected":
            self._connected = False
            self._status_var.set("Not connected -- connect in the Connection tab")
            self._calib_btn.configure(state="disabled")
            self._finder_exp_scale.configure(state="disabled")
            self._finder_gain_scale.configure(state="disabled")
        elif event.kind == "preview_frame":
            frame = pgm_to_array(event.payload["pgm"])
            self._latest_frame = frame
            self._finder_state.update_frame(frame)
            self._show_preview(frame)

    def _show_preview(self, frame: np.ndarray) -> None:
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        gray = frame if frame.ndim == 2 else frame.mean(axis=2).astype(frame.dtype)
        dw, dh, scale, display = downsample_for_display(gray, cw, ch)
        header = f"P5\n{dw} {dh}\n255\n".encode()
        self._photo = tk.PhotoImage(data=header + display.tobytes())
        self._canvas.delete("all")
        xoff = (cw - dw) // 2
        yoff = (ch - dh) // 2
        self._canvas.create_image(xoff, yoff, anchor="nw", image=self._photo)
        # Main camera's own FOV, projected into finder space (see
        # FinderState.main_fov_corners_px) -- shows where the acquisition
        # camera is actually looking within the finder's wider view.
        corners = self._finder_state.main_fov_corners_px()
        if corners is not None:
            points = []
            for row, col in corners:
                points.append(int(col * scale) + xoff)
                points.append(int(row * scale) + yoff)
            self._canvas.create_polygon(points, outline="lime", fill="", width=2)
        # Draw blob marker
        if self._finder_state.blob_found and self._finder_state.last_blob_row is not None:
            bx = int(self._finder_state.last_blob_col * scale) + xoff
            by = int(self._finder_state.last_blob_row * scale) + yoff
            r = 12
            self._canvas.create_oval(bx - r, by - r, bx + r, by + r, outline="red", width=2)
            self._blob_var.set(
                f"ISS blob: ({self._finder_state.last_blob_col:.0f}, {self._finder_state.last_blob_row:.0f}) px"
            )
        else:
            self._blob_var.set("No bright blob detected")
