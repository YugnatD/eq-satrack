"""Multi-point polar-alignment math: given several real (plate-solved)
RA/DEC positions captured while the mount was rotated purely in RA
between each, find the mount's TRUE mechanical rotation axis, and compare
it to the true celestial pole to get an altitude/azimuth correction --
the same principle EKOS's "Polar Alignment Assistant" uses, just without
needing Polaris (or any specific star) in frame: plate solving gives an
absolute RA/DEC for whatever's actually there.

Uses 3 points (not 2): the fit below doesn't need to trust the commanded
rotation angle at all (real motor slip/backlash/timing could make that
imprecise) -- it only needs three real, independently solved sky
positions, which is a properly-conditioned (non-degenerate as long as the
three aren't collinear through the axis) geometric fit rather than one
that leans on an assumed rotation amount.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .angles import angular_separation_deg, equatorial_to_altaz


def _radec_to_unit_vector(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    cos_dec = math.cos(dec)
    return cos_dec * math.cos(ra), cos_dec * math.sin(ra), math.sin(dec)


def _unit_vector_to_radec(v: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = v
    dec_deg = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    ra_deg = math.degrees(math.atan2(y, x)) % 360.0
    return ra_deg, dec_deg


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def fit_rotation_axis(points: list[tuple[float, float]]) -> tuple[float, float]:
    """(axis_ra_deg, axis_dec_deg) of the sphere point equidistant (in
    angle) from all of `points` (each an (ra_deg, dec_deg) pair, real
    plate-solved positions captured at different rotations around the
    mount's own mechanical axis) -- that equidistant point IS the
    mechanical rotation axis. Needs exactly 3 points: the point
    equidistant from three points on a sphere is unique (up to the north/
    south sign ambiguity resolved below), so a least-squares extension to
    more points isn't implemented here -- three well-spread real solves
    are what the calling workflow captures.

    Geometry: convert each point to a 3D unit vector. The axis (also a
    unit vector) satisfies axis . v_i = cos(r) for the same radius r at
    every i, i.e. axis . (v_i - v_j) = 0 for every pair -- so axis is
    perpendicular to both chord vectors (v0-v1) and (v1-v2), i.e. parallel
    to their cross product."""
    if len(points) != 3:
        raise ValueError(f"need exactly 3 points to fit a rotation axis, got {len(points)}")
    v = [_radec_to_unit_vector(ra, dec) for ra, dec in points]
    n = _cross(_sub(v[0], v[1]), _sub(v[1], v[2]))
    norm = _norm(n)
    if norm < 1e-9:
        raise ValueError("the 3 points are degenerate (collinear through the axis, or coincident) -- cannot fit an axis")
    axis = (n[0] / norm, n[1] / norm, n[2] / norm)
    # Two antipodal solutions -- pick the one in the hemisphere the three
    # source points are actually in (a real polar-alignment capture is
    # taken near ONE pole, never spanning both).
    mean_z = sum(p[2] for p in v) / 3.0
    if (axis[2] < 0) != (mean_z < 0):
        axis = (-axis[0], -axis[1], -axis[2])
    return _unit_vector_to_radec(axis)


@dataclass(frozen=True)
class PolarAlignmentResult:
    axis_ra_deg: float  # fitted mechanical rotation-axis RA
    axis_dec_deg: float  # fitted mechanical rotation-axis DEC
    axis_alt_deg: float  # fitted axis's current altitude (should read ~= site latitude)
    axis_az_deg: float  # fitted axis's current azimuth (should read ~= 0, or 180 south of the equator)
    error_deg: float  # total angular separation from the true celestial pole
    error_alt_deg: float  # + : mount's pole is too HIGH -- lower the altitude adjuster
    error_az_deg: float  # + : mount's pole is too far EAST of true north -- rotate the base west (and vice versa)


def polar_alignment_error(axis_ra_deg: float, axis_dec_deg: float, lat_deg: float, lon_deg: float, when) -> PolarAlignmentResult:
    """Converts the fitted mechanical axis (a fixed point in the sky, in
    equatorial coordinates) to ITS current altitude/azimuth at this site
    and compares that to the true celestial pole's -- which is always at
    (altitude=|lat_deg|, azimuth=0 north of the equator or 180 south of
    it), a fixed geometric fact of the site that doesn't depend on `when`
    (only the axis's OWN alt/az conversion does, since alt/az is
    topocentric and Earth keeps rotating under a fixed equatorial point).
    This is exactly the correction the mount's altitude/azimuth adjusters
    need to close: no separate small-angle approximation or extra
    trigonometry beyond the alt/az conversion already used everywhere
    else in this project (am5.angles.equatorial_to_altaz)."""
    axis_az_deg, axis_alt_deg = equatorial_to_altaz(axis_ra_deg, axis_dec_deg, lat_deg, lon_deg, when)
    target_alt_deg = abs(lat_deg)
    target_az_deg = 0.0 if lat_deg >= 0 else 180.0
    error_deg = angular_separation_deg(axis_az_deg, axis_alt_deg, target_az_deg, target_alt_deg)
    error_alt_deg = axis_alt_deg - target_alt_deg
    error_az_deg = ((axis_az_deg - target_az_deg + 180.0) % 360.0) - 180.0
    return PolarAlignmentResult(
        axis_ra_deg=axis_ra_deg, axis_dec_deg=axis_dec_deg,
        axis_alt_deg=axis_alt_deg, axis_az_deg=axis_az_deg,
        error_deg=error_deg, error_alt_deg=error_alt_deg, error_az_deg=error_az_deg,
    )
