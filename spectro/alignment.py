"""Real geometry for a trail that isn't perfectly horizontal on the
sensor -- a slitless grating (Star Analyser) trail's angle is set by how
the grating happens to be rotated relative to the camera, which the
operator identifies by eye (see gui/panels.py's AlignmentPanel: mark
order 0, then trace the spectrum with a click-drag line) rather than
something this app can assume.

One primitive (rotate_sample) does both directions:
- extract_aligned_crop: given a real (or realistically mocked) frame, an
  identified order-0 position, and the measured trail angle, undoes the
  tilt -- produces the same small "already horizontal" working image
  spectro/reduction.py has always assumed, so none of that module needs
  to know about angles at all.
- paste_star_trail: the mock's own use of the same math in the other
  direction, painting a pre-rendered horizontal trail patch into a
  larger canvas at a given position and angle -- what the operator sees
  in the full-frame live view before they've identified anything.
"""

from __future__ import annotations

import numpy as np


def bilinear_sample(source: np.ndarray, x: np.ndarray, y: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Samples `source` (2D) at fractional (x, y) coordinates -- x,y can
    be any shape, arrays of the same shape are returned. Points that fall
    outside `source` (even after floor/ceil) get `fill_value`, not a
    clamped edge pixel -- important here since "no signal" (fill_value=0,
    the mock's own bias level) is a physically correct answer for
    "off the edge of the sensor", clamping to the nearest edge pixel
    wouldn't be."""
    h, w = source.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    fx, fy = x - x0, y - y0
    top = source[y0c, x0c] * (1 - fx) + source[y0c, x1c] * fx
    bottom = source[y1c, x0c] * (1 - fx) + source[y1c, x1c] * fx
    result = top * (1 - fy) + bottom * fy
    return np.where(valid, result, fill_value)


def rotate_sample(
    source: np.ndarray, dst_shape: tuple[int, int], src_anchor: tuple[float, float],
    dst_anchor: tuple[float, float], angle_deg: float, fill_value: float = 0.0,
) -> np.ndarray:
    """Builds an array of `dst_shape` (h, w) by sampling `source`, such
    that `src_anchor` (x, y) in `source` lands at `dst_anchor` (x, y) in
    the output, and a step along the output's own +x axis corresponds to
    a step along `source`'s +x axis rotated by `angle_deg` (degrees,
    counter-clockwise, standard math convention). Bilinear-interpolated;
    see bilinear_sample for the out-of-bounds behavior.

    extract_aligned_crop and paste_star_trail are this same primitive
    used in opposite directions (undoing vs. applying a tilt) -- see
    their own docstrings for which sign of angle_deg each needs and why
    they're exact inverses of each other."""
    h, w = dst_shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    dx, dy = xx - dst_anchor[0], yy - dst_anchor[1]
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    src_x = src_anchor[0] + dx * cos_t + dy * sin_t
    src_y = src_anchor[1] - dx * sin_t + dy * cos_t
    return bilinear_sample(source, src_x, src_y, fill_value)


def extract_aligned_crop(
    image: np.ndarray, order0_xy: tuple[float, float], angle_deg: float,
    crop_shape: tuple[int, int] = (90, 420), local_anchor: tuple[float, float] = (55.0, 45.0),
    fill_value: float = 0.0,
) -> np.ndarray:
    """Crops `image` around `order0_xy` and rotates by -angle_deg so a
    trail that's actually tilted by `angle_deg` on the real sensor comes
    out horizontal in the returned crop -- `local_anchor` is where order
    0 ends up within it, matching spectro/reduction.py's own
    TRAIL_ROW/TRAIL_START_PX assumptions (55, 45) by default so its
    extract_profile etc. need no changes at all."""
    return rotate_sample(image, crop_shape, order0_xy, local_anchor, -angle_deg, fill_value)


def paste_star_trail(
    canvas: np.ndarray, patch: np.ndarray, star_xy: tuple[float, float], angle_deg: float,
    patch_anchor: tuple[float, float] = (55.0, 45.0),
) -> np.ndarray:
    """Inverse of extract_aligned_crop: paints the small, already-
    horizontal `patch` (e.g. from _synthetic_trail_image) into `canvas`
    (typically a full-sensor-sized mock frame) so the star lands at
    `star_xy` and the trail runs off at `angle_deg` from the canvas's own
    +x axis -- what the live full-frame view shows before the operator
    has identified anything. Returns a NEW array (`canvas` isn't modified
    in place) with the patch bilinearly blended in; canvas pixels the
    patch doesn't reach are left as `canvas` already had them (the
    patch's own background/noise floor is discarded, not double-added --
    only signal above roughly canvas's own bias would show visibly
    anyway, but keeping this additive-free avoids brightening the
    surrounding noise floor for no physical reason)."""
    sampled = rotate_sample(patch, canvas.shape, patch_anchor, star_xy, angle_deg, fill_value=np.nan)
    mask = np.isfinite(sampled)
    return np.where(mask, sampled, canvas)


def angle_from_points(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """Angle in degrees (counter-clockwise from +x, standard math/screen-
    y-down convention matching rotate_sample) of the line from p0 to p1
    -- what AlignmentPanel's click-drag trace line measures."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return float(np.degrees(np.arctan2(-dy, dx)))
