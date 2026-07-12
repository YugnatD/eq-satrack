"""In-process mock of an ASI290MC-like camera, for developing/testing the
capture pipeline without hardware — same role as am5/mock_mount.py.

Generates synthetic RAW8 frames. Two modes:

- Untethered (nothing has ever called set_sky_context): a bright blob
  sweeps across the ROI on a fixed 2s loop, just so a standalone
  camera-only session (no mount connected) still shows something moving.
- Tethered to a mount (the GUI's App wires this once both workers are
  connected, see am5/gui/app.py): a real star field -- the bundled
  Hipparcos-derived catalog, assets/bright_stars.npz, ~98700 stars down to
  Vmag 9.5 -- is rendered relative to wherever the mount is actually
  pointed (the "boresight"), brightness scaled by each star's real
  magnitude AND by exposure/gain (see EXPOSURE_REFERENCE_US below) --
  exactly like a real sensor, a longer exposure integrates more signal and
  reveals fainter catalog stars, not just a brighter version of the same
  ones. The ISS is rendered at its offset from that boresight. Both are
  computed fresh from real RA/DEC every frame (a tangent-plane projection,
  no field rotation modeled), not accumulated from an abstract pixel
  offset -- so panning/jogging the mount correctly pans the simulated
  field, and pointing at a real star (e.g. Vega) actually shows that star,
  at the plate scale of whatever optical train is configured in the
  Exposure calc tab (see set_sky_context's caller in am5/gui/panels.py's
  TransitPanel._on_camera_connect_click). With a narrow field of view
  (typical for ISS imaging -- long focal length), don't expect a crowded
  field at a quick exposure: real star density this bright is low enough
  that most narrow pointings show few or no companion stars at a glance,
  matching what a real ISS rig would actually see -- it's specifically
  winding the exposure slider up that should reveal more.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np

from am5.angles import circular_diff_deg

# Matches this project's own "typical ISS rig" default (200mm aperture,
# 1000mm focal length, 2.9um pixels -- see am5/optics.py's ExposurePanel
# defaults), so the mock's field of view is consistent with what the
# exposure calculator assumes elsewhere in the app. Overridden with the
# actually-configured train when connected via the GUI (see module
# docstring).
DEFAULT_ARCSEC_PER_PIXEL = 0.6

# ~98700 real stars (Hipparcos, Vmag < 9.5 -- near the catalogue's own
# completeness limit) -- see assets/bright_stars.LICENSE.txt for provenance.
STAR_CATALOG_PATH = Path(__file__).resolve().parent.parent / "assets" / "bright_stars.npz"

# Background stars are rendered dimmer than the ISS and scaled by their
# real magnitude (5 mag = 100x flux, the standard astronomical scale) -- a
# real ISS capture uses a sub-millisecond exposure specifically because the
# ISS is so bright, so only the brightest stars register at all, faintly.
# STAR_MAG0_PEAK is deliberately a bit under ISS_PEAK_VALUE so the ISS is
# always the brightest thing in frame even next to a hypothetical mag-0 star.
STAR_MAG0_PEAK = 180.0
STAR_MIN_VISIBLE_PEAK = 3.0  # skip rendering (and the gaussian-patch cost) below this -- indistinguishable from noise anyway
STAR_SIGMA_PX = 1.2
ISS_PEAK_VALUE = 200.0
ISS_SIGMA_PX = 3.0

# Reference exposure at which exposure_scale == 1.0 -- matches this class's
# own __init__ default, so brightness at default settings is unchanged from
# before exposure had any effect. Real sensors integrate signal roughly
# linearly with exposure time, unlike gain_scale below (which is a coarse
# stand-in, not true dB-based gain math) -- so this is a direct ratio, not
# another capped curve.
EXPOSURE_REFERENCE_US = 1000.0

_star_catalog_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def _load_star_catalog() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ra_deg, dec_deg, magnitude) arrays, loaded once and cached at
    module level -- ~98700 stars, still cheap to keep resident (a few MB)."""
    global _star_catalog_cache
    if _star_catalog_cache is None:
        with np.load(STAR_CATALOG_PATH) as data:
            _star_catalog_cache = (
                data["ra_deg"].astype(np.float64), data["dec_deg"].astype(np.float64), data["magnitude"].astype(np.float64),
            )
    return _star_catalog_cache


class MockAsiCamera:
    """Duck-type compatible with camera.asi_camera.AsiCamera."""

    def __init__(self, seed: int | None = None, plate_scale_arcsec_per_px: float = DEFAULT_ARCSEC_PER_PIXEL):
        self._rng = np.random.default_rng(seed)
        self._x, self._y = 0, 0
        self._width, self._height = 640, 480
        self._exposure_us = 1000
        self._gain = 300
        self._streaming = False
        self._t0 = 0.0
        self._opened = False
        self._plate_scale = plate_scale_arcsec_per_px
        self._sky_lock = threading.Lock()
        self._boresight_radec: tuple[float, float] | None = None  # (ra_deg, dec_deg)
        self._target_radec: tuple[float, float] | None = None  # (ra_deg, dec_deg) -- the ISS

    def open(self) -> None:
        self._opened = True
        self._t0 = time.monotonic()

    def close(self) -> None:
        self._streaming = False
        self._opened = False

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        self._x, self._y, self._width, self._height = x, y, width, height

    def set_exposure_us(self, microseconds: int) -> None:
        self._exposure_us = max(1, int(microseconds))

    def set_gain(self, gain: int) -> None:
        self._gain = int(gain)

    def set_sky_context(self, boresight_ra_deg: float, boresight_dec_deg: float, target_ra_deg: float, target_dec_deg: float) -> None:
        """Called from the CameraWorker thread (see camera/worker.py),
        fed with the mount's actual current RA/DEC (boresight) and the
        real or reference target RA/DEC (the ISS, or the training
        reference point -- see am5/gui/app.py) on every mount position
        update. Thread-safe: written here, read from read_frame() on the
        CameraWorker's own thread."""
        with self._sky_lock:
            self._boresight_radec = (boresight_ra_deg, boresight_dec_deg)
            self._target_radec = (target_ra_deg, target_dec_deg)

    def get_controls(self) -> dict:
        return {
            "Exposure": {"Name": "Exposure", "MinValue": 32, "MaxValue": 2_000_000_000, "DefaultValue": 10000},
            "Gain": {"Name": "Gain", "MinValue": 0, "MaxValue": 570, "DefaultValue": 300},
        }

    def start_streaming(self) -> None:
        self._streaming = True
        self._t0 = time.monotonic()

    def stop_streaming(self) -> None:
        self._streaming = False

    def _tangent_offset_arcsec(self, ra_deg: float, dec_deg: float, from_ra_deg: float, from_dec_deg: float) -> tuple[float, float]:
        """(dx_arcsec, dy_arcsec) of (ra_deg, dec_deg) relative to
        (from_ra_deg, from_dec_deg) in the tangent plane -- no field
        rotation modeled, fine at the narrow angular scales here."""
        cos_dec = math.cos(math.radians(from_dec_deg))
        dx = circular_diff_deg(ra_deg, from_ra_deg) * cos_dec * 3600.0
        dy = (dec_deg - from_dec_deg) * 3600.0
        return dx, dy

    def _render_stars(self, frame: np.ndarray, boresight_ra: float, boresight_dec: float, gain_scale: float) -> None:
        """Vectorized over the whole ~98700-star catalog: compute every
        star's tangent-plane pixel position at once (numpy, not a Python
        loop over 98700 stars), filter to the handful actually in view, then
        rasterize just those -- keeps this real-time regardless of catalog
        depth, since the per-frame cost that scales with star count is a
        few vectorized array ops, not a Python loop."""
        star_ra, star_dec, star_mag = _load_star_catalog()
        cos_dec = math.cos(math.radians(boresight_dec))
        dx = circular_diff_deg(star_ra, boresight_ra) * cos_dec * 3600.0
        dy = (star_dec - boresight_dec) * 3600.0
        star_x = self._width / 2.0 + dx / self._plate_scale
        star_y = self._height / 2.0 - dy / self._plate_scale
        margin = 5.0
        in_view = (star_x >= -margin) & (star_x <= self._width + margin) & (star_y >= -margin) & (star_y <= self._height + margin)
        for x, y, mag in zip(star_x[in_view], star_y[in_view], star_mag[in_view]):
            peak = STAR_MAG0_PEAK * (10.0 ** (-0.4 * mag)) * gain_scale
            if peak < STAR_MIN_VISIBLE_PEAK:
                continue
            self._draw_point(frame, float(x), float(y), float(peak), STAR_SIGMA_PX, margin=margin)

    def _draw_point(self, frame: np.ndarray, x: float, y: float, peak: float, sigma: float, margin: float) -> None:
        if not (-margin <= x <= self._width + margin and -margin <= y <= self._height + margin):
            return
        # Only rasterize a small box around the point -- cheap even with
        # many background stars, unlike a full-frame gaussian per star.
        radius = int(math.ceil(sigma * 4))
        x0, x1 = max(0, int(x) - radius), min(self._width, int(x) + radius + 1)
        y0, y1 = max(0, int(y) - radius), min(self._height, int(y) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        frame[y0:y1, x0:x1] += peak * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)))

    def read_frame(self, timeout_ms: int = 2000) -> np.ndarray:
        # Pace frames by the configured exposure, like a real sensor's frame
        # interval — lets the worker measure a believable fps.
        time.sleep(max(self._exposure_us / 1_000_000.0, 0.002))

        elapsed = time.monotonic() - self._t0
        # Both gain and exposure make faint stars visible, like a real
        # sensor -- previously only gain had any effect, which left
        # exposure as a dead control for "how many stars show up".
        exposure_scale = self._exposure_us / EXPOSURE_REFERENCE_US
        gain_scale = (1.0 + self._gain / 570.0) * exposure_scale
        noise_sigma = 6.0 / max(1.0 + self._gain / 570.0, 0.1)
        frame = self._rng.normal(20.0, noise_sigma, size=(self._height, self._width))

        with self._sky_lock:
            boresight, target = self._boresight_radec, self._target_radec

        if boresight is not None:
            boresight_ra, boresight_dec = boresight
            self._render_stars(frame, boresight_ra, boresight_dec, gain_scale)

            if target is not None:
                target_ra, target_dec = target
                dx, dy = self._tangent_offset_arcsec(target_ra, target_dec, boresight_ra, boresight_dec)
                # Small gaussian jitter stands in for seeing/guiding wobble
                # so a perfectly-tracked target isn't an unnaturally frozen
                # pixel.
                iss_x = self._width / 2.0 + dx / self._plate_scale + self._rng.normal(0.0, 0.5)
                iss_y = self._height / 2.0 - dy / self._plate_scale + self._rng.normal(0.0, 0.5)
                self._draw_point(frame, iss_x, iss_y, ISS_PEAK_VALUE * gain_scale, ISS_SIGMA_PX, margin=20.0)
        else:
            # No mount tethered yet -- fixed demo sweep so a standalone
            # camera-only session still shows motion.
            sweep_period_s = 2.0
            phase = (elapsed % sweep_period_s) / sweep_period_s
            blob_x = phase * self._width
            blob_y = self._height / 2.0
            self._draw_point(frame, blob_x, blob_y, ISS_PEAK_VALUE * gain_scale, ISS_SIGMA_PX, margin=20.0)

        return np.clip(frame, 0, 255).astype(np.uint8)

    def bayer_pattern_ser_colour_id(self) -> int:
        return 8  # pretend RGGB, matching the ASI290MC's colour sensor

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_color(self) -> bool:
        return True

    @property
    def bit_depth(self) -> int:
        return 8
