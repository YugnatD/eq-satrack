"""Camera-based closed-loop guiding: find the ISS in a frame and convert
that into a sky-plane correction, so the same trigger_perp_pulse()/
adjust_delta_t() an operator already drives by hand (see am5/tracker.py's
LiveOffsets) can be driven automatically from the live image instead.

Two independent pieces:
- detect_brightest_blob(): the ISS is far brighter than sky/background in a
  short exposure, so a simple threshold + intensity-weighted centroid is
  enough -- no need for real segmentation/multi-object tracking.
- GuidingCalibration: maps a detected pixel offset back to a sky-plane
  (RA, DEC) offset. This mapping depends on the camera's mounting
  rotation relative to the sky, which is NOT known in advance (unlike the
  mount's own RA/DEC axes, which characterize.py/calibrate_directions
  already determine) -- it must be measured per-session, the same
  rationale as axis-sign calibration but for the camera instead of the
  mount.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.registration import phase_cross_correlation


@dataclass(frozen=True)
class BlobDetection:
    found: bool
    centroid_x: float  # pixels, frame-relative (0,0 = top-left, matches array indexing)
    centroid_y: float
    peak_value: float
    pixel_count: int  # size of the thresholded region -- a too-small blob is probably noise, not the ISS


def detect_brightest_blob(frame: np.ndarray, threshold_sigma: float = 6.0, min_pixels: int = 2) -> BlobDetection:
    """Thresholds at background_mean + threshold_sigma*background_std, then
    takes the intensity-weighted centroid of every pixel above that. Doesn't
    attempt to separate multiple bright objects -- fine for a short/narrow
    exposure where the ISS is the only thing that bright, not fine for a
    star field with comparable-brightness stars in frame."""
    frame = frame.astype(np.float64)
    mean, std = float(frame.mean()), float(frame.std())
    threshold = mean + threshold_sigma * std
    mask = frame > threshold
    pixel_count = int(mask.sum())
    if pixel_count < min_pixels:
        return BlobDetection(found=False, centroid_x=0.0, centroid_y=0.0, peak_value=float(frame.max()), pixel_count=pixel_count)
    yy, xx = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
    weights = frame * mask
    total = float(weights.sum())
    cx = float((xx * weights).sum() / total)
    cy = float((yy * weights).sum() / total)
    return BlobDetection(found=True, centroid_x=cx, centroid_y=cy, peak_value=float(frame.max()), pixel_count=pixel_count)


@dataclass(frozen=True)
class GuidingCalibration:
    """Linear map between a sky-plane offset (d_ra_arcsec, d_dec_arcsec --
    RA already scaled by cos(dec), i.e. a tangent-plane offset, not raw
    RA degrees) and a camera-pixel offset (dx_px, dy_px), measured via two
    known-axis nudges (see calibrate_from_nudges). Captures whatever
    rotation/mirroring the camera happens to be mounted at -- there's no
    way to assume this in software, it has to be measured."""

    px_per_ra_arcsec_x: float
    px_per_ra_arcsec_y: float
    px_per_dec_arcsec_x: float
    px_per_dec_arcsec_y: float

    def sky_to_pixel(self, d_ra_arcsec: float, d_dec_arcsec: float) -> tuple[float, float]:
        dx = self.px_per_ra_arcsec_x * d_ra_arcsec + self.px_per_dec_arcsec_x * d_dec_arcsec
        dy = self.px_per_ra_arcsec_y * d_ra_arcsec + self.px_per_dec_arcsec_y * d_dec_arcsec
        return dx, dy

    def pixel_to_sky(self, dx_px: float, dy_px: float) -> tuple[float, float]:
        a, b, c, d = self.px_per_ra_arcsec_x, self.px_per_dec_arcsec_x, self.px_per_ra_arcsec_y, self.px_per_dec_arcsec_y
        det = a * d - b * c
        if abs(det) < 1e-9:
            raise ValueError("degenerate guiding calibration (RA and DEC nudges produced the same pixel direction)")
        d_ra_arcsec = (d * dx_px - b * dy_px) / det
        d_dec_arcsec = (-c * dx_px + a * dy_px) / det
        return d_ra_arcsec, d_dec_arcsec

    @property
    def arcsec_per_pixel(self) -> float:
        """Rough scale for display purposes -- sqrt of the matrix
        determinant's magnitude gives the area-scaling factor, whose square
        root is a representative linear scale when RA/DEC nudges produced
        roughly perpendicular pixel directions (the expected case)."""
        det = self.px_per_ra_arcsec_x * self.px_per_dec_arcsec_y - self.px_per_dec_arcsec_x * self.px_per_ra_arcsec_y
        return 1.0 / max(abs(det), 1e-12) ** 0.5


def measure_frame_shift(
    reference_frame: np.ndarray, live_frame: np.ndarray, downsample: int = 4, max_error: float = 0.3,
) -> tuple[float, float] | None:
    """How far the star field in live_frame has apparently shifted since
    reference_frame, via FFT phase correlation -- the cheap (sub-100ms),
    no-plate-solve-needed measurement am5.polar_alignment's
    axis_radec_from_frame_shift needs on every PAA live-estimate refresh
    tick (am5/gui/panels.py's AlignmentPanel, 5Hz), instead of a real
    multi-second astrometry.net solve.

    Both frames are downsampled by integer-stride slicing first (same
    technique and same rationale as camera/finder.py's
    downsample_for_display -- skimage.transform.resize's interpolation
    measurably freezes the Tk main thread at this project's real sensor
    sizes and refresh rates; a plain stride is ~free and phase
    correlation doesn't need the smoothing).

    Returns (delta_col, delta_row) in FULL-RESOLUTION pixels, in this
    project's own image convention (column increases right/east, row
    increases down -- see project_radec_to_pixel's own docstring), ready
    to hand directly to axis_radec_from_frame_shift. Returns None if
    phase_cross_correlation's own normalized error exceeds max_error
    (field out of frame, cloud, nothing left to correlate against) rather
    than silently returning a meaningless number -- confirmed by manual
    testing this session that normalization=None (the default
    normalization='phase' gives an error metric close to 1.0 for BOTH a
    clean match and pure noise in this project's installed scikit-image
    0.25.2, useless as a confidence signal) discriminates clearly: ~1e-7
    for a true match, several tenths to ~1.0 for an unrelated pair.

    skimage.registration.phase_cross_correlation's own returned shift is
    the NEGATIVE of the actual content displacement (verified by manual
    script this session: phase_cross_correlation(reference, moving) where
    moving = shift(reference, +delta) returns ~= -delta) and is in
    (row, col) order (skimage's own array-axis convention, not this
    project's [x, y] pixel convention) -- both corrected for here before
    returning."""
    ref = np.asarray(reference_frame)[::downsample, ::downsample].astype(np.float64)
    live = np.asarray(live_frame)[::downsample, ::downsample].astype(np.float64)
    min_h = min(ref.shape[0], live.shape[0])
    min_w = min(ref.shape[1], live.shape[1])
    ref, live = ref[:min_h, :min_w], live[:min_h, :min_w]

    shift, error, _diffphase = phase_cross_correlation(ref, live, upsample_factor=10, normalization=None)
    if error > max_error:
        return None
    delta_row_px = -shift[0] * downsample
    delta_col_px = -shift[1] * downsample
    return delta_col_px, delta_row_px


def calibrate_from_nudges(
    d_ra_arcsec: float, ra_nudge_dx_px: float, ra_nudge_dy_px: float,
    d_dec_arcsec: float, dec_nudge_dx_px: float, dec_nudge_dy_px: float,
) -> GuidingCalibration:
    """Builds a GuidingCalibration from two measured nudges: a pure-RA
    nudge of `d_ra_arcsec` that moved the blob by (ra_nudge_dx_px,
    ra_nudge_dy_px), and a pure-DEC nudge of `d_dec_arcsec` that moved it
    by (dec_nudge_dx_px, dec_nudge_dy_px). `d_ra_arcsec`/`d_dec_arcsec` must
    be measured from the mount's own actual RA/DEC change (:GMEQ# before
    and after), not assumed from the commanded rate*duration -- mechanical
    ramp lag makes that unreliable (same reasoning as
    am5.tracker.calibrate_directions)."""
    if abs(d_ra_arcsec) < 1e-6 or abs(d_dec_arcsec) < 1e-6:
        raise ValueError("nudge produced no measurable sky motion -- check the mount actually moved")
    return GuidingCalibration(
        px_per_ra_arcsec_x=ra_nudge_dx_px / d_ra_arcsec,
        px_per_ra_arcsec_y=ra_nudge_dy_px / d_ra_arcsec,
        px_per_dec_arcsec_x=dec_nudge_dx_px / d_dec_arcsec,
        px_per_dec_arcsec_y=dec_nudge_dy_px / d_dec_arcsec,
    )
