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
# 1.5, not the original 3.0 -- at default gain/exposure this puts the
# visible magnitude limit around ~5.7 instead of ~4.9 (confirmed: with the
# old 3.0 threshold, a wide-FOV finder camera and the narrow main camera
# showed almost the same star count for most sky positions, since the
# catalogue is sparse above mag~5 -- the FOV difference only becomes
# visible once faint-enough stars are allowed through). Kept above the
# noise floor (~0.4-1.1x noise_sigma depending on gain) rather than
# pushed all the way down to the catalogue's real mag~9.5 depth, which
# would put most of the newly-revealed stars indistinguishable from noise.
STAR_MIN_VISIBLE_PEAK = 1.5  # skip rendering (and the gaussian-patch cost) below this -- indistinguishable from noise anyway
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

# Real minimum frame interval regardless of configured exposure -- see
# read_frame's own comment for why this exists (CPU/thermal, not just a
# nicety). ~33fps ceiling, still plenty smooth for a mock preview.
MIN_FRAME_INTERVAL_S = 0.03

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

    def __init__(
        self, seed: int | None = None, plate_scale_arcsec_per_px: float = DEFAULT_ARCSEC_PER_PIXEL,
        bit_depth: int = 8,
        sensor_width: int = 640, sensor_height: int = 480,
    ):
        self._rng = np.random.default_rng(seed)
        self._x, self._y = 0, 0
        self._width, self._height = sensor_width, sensor_height
        self._sensor_width, self._sensor_height = sensor_width, sensor_height
        self._exposure_us = 1000
        self._gain = 300
        self._bit_depth = bit_depth
        self._streaming = False
        self._t0 = 0.0
        self._opened = False
        self._plate_scale = plate_scale_arcsec_per_px
        self._sky_lock = threading.Lock()
        self._boresight_radec: tuple[float, float] | None = None  # (ra_deg, dec_deg)
        self._target_radec: tuple[float, float] | None = None  # (ra_deg, dec_deg) -- the ISS
        # Precomputed once per (width, height) instead of drawing a fresh
        # full-frame gaussian every read_frame() call -- rng.normal() over
        # the finder's 3840x2160 frame alone measured ~93ms, which pinned a
        # full CPU core (confirmed via psutil: ~97.5% utilization) and was
        # the real cause of a reported sustained 100°C CPU temperature with
        # two mock cameras running. Each frame instead slices a random
        # window of this pool (~9ms) -- still a different-looking pattern
        # every frame (the offset moves), just not independently drawn.
        self._noise_pool: np.ndarray | None = None
        self._noise_pool_dims: tuple[int, int] | None = None

    def open(self) -> None:
        self._opened = True
        self._t0 = time.monotonic()

    def close(self) -> None:
        self._streaming = False
        self._opened = False

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        # Mirrors the real ASI SDK's width-multiple-of-8/height-multiple-of-2
        # rounding (see AsiCamera.set_roi) so ROI behavior matches real
        # hardware even when developing against the mock.
        width = max(8, (width // 8) * 8)
        height = max(2, (height // 2) * 2)
        self._x, self._y, self._width, self._height = x, y, width, height

    def set_bit_depth(self, bit_depth: int) -> None:
        if bit_depth not in (8, 16):
            raise ValueError(f"unsupported bit depth {bit_depth!r} (must be 8 or 16)")
        self._bit_depth = bit_depth

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

    # How far the random window can roam inside the pool -- also what makes
    # consecutive frames look like independent noise despite sharing a pool.
    _NOISE_POOL_MARGIN = 64

    def _draw_noise(self, sigma: float) -> np.ndarray:
        """A (height, width) background-noise array, mean 20.0 -- see the
        _noise_pool comment in __init__ for why this doesn't call
        rng.normal() fresh every frame."""
        dims = (self._height, self._width)
        if self._noise_pool is None or self._noise_pool_dims != dims:
            pool_h = self._height + self._NOISE_POOL_MARGIN
            pool_w = self._width + self._NOISE_POOL_MARGIN
            self._noise_pool = self._rng.standard_normal(size=(pool_h, pool_w)).astype(np.float32)
            self._noise_pool_dims = dims
        oy = self._rng.integers(0, self._NOISE_POOL_MARGIN + 1)
        ox = self._rng.integers(0, self._NOISE_POOL_MARGIN + 1)
        window = self._noise_pool[oy:oy + self._height, ox:ox + self._width]
        return window * sigma + 20.0

    def _render_float_frame(self) -> np.ndarray:
        """Builds one frame of the synthetic scene at full float precision,
        unclamped to any particular bit depth -- read_frame clamps this to
        either 8-bit or the sensor's real 12-bit ADC range depending on
        the currently configured bit_depth (see set_bit_depth)."""
        elapsed = time.monotonic() - self._t0
        # Both gain and exposure make faint stars visible, like a real
        # sensor -- previously only gain had any effect, which left
        # exposure as a dead control for "how many stars show up".
        exposure_scale = self._exposure_us / EXPOSURE_REFERENCE_US
        gain_scale = (1.0 + self._gain / 570.0) * exposure_scale
        noise_sigma = 6.0 / max(1.0 + self._gain / 570.0, 0.1)
        frame = self._draw_noise(noise_sigma)

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

        return frame

    def read_frame(self, timeout_ms: int = 2000) -> np.ndarray:
        # Pace frames by the configured exposure, like a real sensor's frame
        # interval — lets the worker measure a believable fps. Floored at
        # MIN_FRAME_INTERVAL_S (not the old 2ms), which caps this at
        # roughly 30fps regardless of how short the configured exposure is
        # -- _render_float_frame's own cost (a fresh full-frame gaussian
        # noise draw + star rendering every call, ~35-120ms measured on
        # this machine depending on sensor size) means CameraWorker's read
        # loop (see camera/worker.py's _run, which just calls read_frame
        # back-to-back with no throttle of its own) would otherwise pin a
        # full CPU core continuously per connected mock camera -- confirmed
        # as the cause of a real sustained 100°C CPU temperature with two
        # mock cameras (main + finder) connected at once. A real ASI290MC/
        # ASI678MM wouldn't sustain anywhere near 500fps in practice either,
        # so this is a more realistic ceiling, not just a workaround.
        time.sleep(max(self._exposure_us / 1_000_000.0, MIN_FRAME_INTERVAL_S))
        frame = self._render_float_frame()
        if self._bit_depth == 16:
            return np.clip(frame * (4095.0 / 255.0), 0, 4095).astype(np.uint16)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def get_dropped_frames(self) -> int:
        # No real ring buffer to overflow -- read_frame() paces itself by
        # sleeping for the configured exposure, so the mock can't fall
        # behind its own frame source the way a real sensor can.
        return 0

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
        return self._bit_depth
