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
