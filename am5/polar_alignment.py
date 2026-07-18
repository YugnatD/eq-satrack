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

import numpy as np

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


def true_pole_radec(lat_deg: float) -> tuple[float, float]:
    """The true celestial pole's own equatorial position -- fixed by
    definition (it's Earth's rotation axis, essentially motionless in the
    equatorial frame over the timescales this app cares about), unlike its
    ALTITUDE/AZIMUTH which is topocentric and changes as Earth turns
    underneath it (see polar_alignment_error, which works in alt/az
    instead for exactly that reason). RA is meaningless at the pole itself
    -- 0.0 here is an arbitrary placeholder, safe because
    tangent_plane_offset_arcsec/project_radec_to_pixel are verified to
    give an RA-independent result when dec_deg is +-90 (see
    tests/test_polar_alignment.py)."""
    return 0.0, 90.0 if lat_deg >= 0 else -90.0


def correction_triangle_radec(
    axis_ra_deg: float, axis_dec_deg: float, lat_deg: float, lon_deg: float, when,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two RA/DEC waypoints tracing the KStars-style polar-alignment
    correction path from the fitted axis to the true pole, split into a
    pure-azimuth leg then a pure-altitude leg -- a real mount's two
    adjusters only ever move independently, so that's the actual sky path
    (and the directions worth drawing on a live view), not a straight
    line: (az_corrected_point, true_pole_point). az_corrected_point is
    where the axis would land if ONLY the azimuth error were corrected
    (same altitude as the axis, azimuth equal to the true pole's); the
    second leg from there to true_pole_point is then a pure-altitude
    change by construction, no separate computation needed. Doesn't need
    polar_alignment_error's own error_az_deg as an input -- the target
    azimuth (0 north of the equator, 180 south of it) is a fixed fact of
    the site, the same one true_pole_radec/polar_alignment_error use.

    There's no fixed image/sky direction for "azimuth" or "altitude" --
    the local correspondence between (RA, DEC) and (az, alt) steps rotates
    with parallactic angle through the night, AND (being this close to the
    pole by construction -- that's where PAA points) is strongly
    nonlinear over anything but a tiny step, so a single linear
    extrapolation by the full error measurably drifts off the true
    constant-altitude path (confirmed: ~0.2 deg altitude drift on a
    0.84 deg azimuth correction in testing). This instead runs a proper
    multi-step Newton iteration -- solving az(ra,dec)=target_az AND
    alt(ra,dec)=alt0 exactly (to numerical precision), re-linearizing
    equatorial_to_altaz's local Jacobian at each step -- rather than a
    hand-derived closed form (easy to get a sign wrong in -- see
    project_radec_to_pixel's own docstring on why THAT was verified from
    scratch instead of guessed)."""
    target_az_deg = 0.0 if lat_deg >= 0 else 180.0
    az0, alt0 = equatorial_to_altaz(axis_ra_deg, axis_dec_deg, lat_deg, lon_deg, when)
    ra, dec = axis_ra_deg, axis_dec_deg
    eps_deg = 0.005
    for _ in range(12):
        az, alt = equatorial_to_altaz(ra, dec, lat_deg, lon_deg, when)
        f_az = ((az - target_az_deg + 180.0) % 360.0) - 180.0
        f_alt = alt - alt0
        if abs(f_az) < 1e-7 and abs(f_alt) < 1e-7:
            break
        az_dra, alt_dra = equatorial_to_altaz(ra + eps_deg, dec, lat_deg, lon_deg, when)
        az_ddec, alt_ddec = equatorial_to_altaz(ra, dec + eps_deg, lat_deg, lon_deg, when)
        d_az_d_ra = (((az_dra - az) + 180.0) % 360.0 - 180.0) / eps_deg
        d_alt_d_ra = (alt_dra - alt) / eps_deg
        d_az_d_dec = (((az_ddec - az) + 180.0) % 360.0 - 180.0) / eps_deg
        d_alt_d_dec = (alt_ddec - alt) / eps_deg
        det = d_az_d_ra * d_alt_d_dec - d_az_d_dec * d_alt_d_ra
        if abs(det) < 1e-9:
            raise ValueError("degenerate local alt/az<->RA/DEC Jacobian -- cannot resolve a correction direction")
        d_ra = (d_alt_d_dec * (-f_az) - d_az_d_dec * (-f_alt)) / det
        d_dec = (-d_alt_d_ra * (-f_az) + d_az_d_ra * (-f_alt)) / det
        ra += d_ra
        dec += d_dec

    return (ra, dec), true_pole_radec(lat_deg)


def tangent_plane_offset_arcsec(ra_deg, dec_deg, center_ra_deg: float, center_dec_deg: float):
    """Standard-coordinates (xi, eta) gnomonic/TAN tangent-plane offset of
    (ra_deg, dec_deg) from (center_ra_deg, center_dec_deg), in arcsec.
    Deliberately NOT the flat small-angle approximation used elsewhere in
    this project for small on-sky offsets (e.g. circular_diff_deg-based
    ones) -- that approximation is ill-defined right where
    project_radec_to_pixel below most needs it to work: placing the true
    celestial pole, where RA has no meaning -- and was confirmed (see
    camera/mock_camera.py's _render_stars, which used to have its own
    flat approximation) to measurably distort simulated star positions
    within about a degree of the pole (hundreds of arcsec of error,
    exactly where PAA points). This is the same TAN/gnomonic projection
    real plate solvers' own WCS uses, so it reduces to the flat
    approximation for small offsets but stays correct at any separation,
    including exactly at dec=+-90 (verified RA-independent there against
    astropy.wcs, see tests/test_polar_alignment.py).

    Built on numpy (not the math module) throughout so it works
    unchanged on both plain floats (the polar-alignment overlay's use
    case) and whole coordinate arrays (camera/mock_camera.py's
    _render_stars, vectorized over its full star catalog every frame)."""
    ra0, dec0 = np.radians(center_ra_deg), np.radians(center_dec_deg)
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    d_ra = ra - ra0
    denom = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(d_ra)
    xi = np.cos(dec) * np.sin(d_ra) / denom
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(d_ra)) / denom
    return np.degrees(xi) * 3600.0, np.degrees(eta) * 3600.0


def project_radec_to_pixel(
    ra_deg: float, dec_deg: float, center_ra_deg: float, center_dec_deg: float,
    pixel_scale_arcsec: float, rotation_deg: float, flip_parity: bool = False,
) -> tuple[float, float]:
    """(delta_col, delta_row) pixel offset of (ra_deg, dec_deg) from a
    solved frame's own centre, given that solve's pixel_scale_arcsec and
    field_rotation_deg (SolveResult's own fields, camera/platesolve.py) --
    delta_col/delta_row are in this project's own image convention (column
    increases right/east, row increases down; see
    camera/finder.py's downsample_for_display and camera/mock_camera.py's
    _render_stars, which both use exactly this layout), so the result can
    be added directly to a frame-centre pixel position for drawing.

    rotation_deg must be extracted the same way camera/platesolve.py's
    _parse_astap_ini/_parse_astrometry_wcs already do
    (atan2(CD1_2, CD1_1)) -- this function's sign conventions were derived
    and verified specifically against that extraction, by round-tripping
    through astropy.wcs with a CD matrix constructed to match this
    project's own pixel layout (see tests/test_polar_alignment.py) --
    getting this wrong would make the polar-alignment overlay point the
    WRONG direction, worse than not drawing it at all, hence the
    from-scratch verification rather than a guessed formula.

    flip_parity: a real optical path can be mirrored relative to this
    project's own assumed pixel layout (a plain rotation, no reflection --
    the CD matrix built in tests/test_polar_alignment.py's own
    _wcs_matching_this_projects_own_pixel_convention always has a negative
    determinant). Confirmed on real hardware: our actual finder camera's
    solved WCS has a POSITIVE determinant (det(CD) ~= +2.18e-7, verified
    against the real solved point3_attempt5.fits from a live PAA run) --
    the opposite parity from what this function assumed, which was
    reported live as the correction-overlay arrows pointing a direction
    that didn't match reality when the operator tried to follow them
    physically. SolveResult.flip_parity (camera/platesolve.py) is set from
    the ACTUAL solved CD matrix's determinant sign, not guessed or
    hardcoded, and should always be threaded through from there. Derived
    and verified the same from-scratch way as the base (non-flipped)
    formula: round-tripped against astropy.wcs with a determinant-flipped
    CD matrix across several rotations (see
    test_project_radec_to_pixel_matches_astropy_wcs_for_a_flipped_parity
    in tests/test_polar_alignment.py) -- negating north_arcsec before the
    otherwise-unchanged formula reproduces astropy's own pixel mapping to
    within floating-point precision at every rotation tested, not just
    rot=0."""
    east_arcsec, north_arcsec = tangent_plane_offset_arcsec(ra_deg, dec_deg, center_ra_deg, center_dec_deg)
    if flip_parity:
        north_arcsec = -north_arcsec
    rot = math.radians(rotation_deg)
    delta_col = (east_arcsec * math.cos(rot) + north_arcsec * math.sin(rot)) / pixel_scale_arcsec
    delta_row = (east_arcsec * math.sin(rot) - north_arcsec * math.cos(rot)) / pixel_scale_arcsec
    return delta_col, delta_row


def offset_radec_by_east_north(ra_deg: float, dec_deg: float, east_arcsec: float, north_arcsec: float) -> tuple[float, float]:
    """(ra_deg, dec_deg) obtained by moving (ra_deg, dec_deg) by
    (east_arcsec, north_arcsec) on the sky. The FLAT/small-angle inverse
    of tangent_plane_offset_arcsec -- deliberately NOT the exact gnomonic
    inverse that function's own docstring insists on for ITS use cases
    (large separations near the pole, where the flat approximation
    measurably distorts positions). Here the offsets are always small
    (PAA's live correlation-based estimate, am5/gui/panels.py's
    AlignmentPanel -- a fraction of a degree to at most a couple degrees
    of alt/az adjustment), and a LOCAL linearization at (ra_deg, dec_deg)
    is accurate to a small fraction of an arcsec at that scale regardless
    of how close (ra_deg, dec_deg) itself is to the pole (the curvature
    that matters for accuracy is over the SIZE of the offset being
    applied, not the absolute declination) -- verified by round-tripping
    through tangent_plane_offset_arcsec itself in
    tests/test_polar_alignment.py, not against astropy.wcs (this is an
    intentionally different, approximate contract from that function's
    own exact one)."""
    dec_rad = math.radians(dec_deg)
    new_dec_deg = dec_deg + north_arcsec / 3600.0
    new_ra_deg = ra_deg + (east_arcsec / 3600.0) / max(math.cos(dec_rad), 1e-6)
    return new_ra_deg % 360.0, new_dec_deg


def axis_radec_from_frame_shift(
    prev_axis_ra_deg: float, prev_axis_dec_deg: float, delta_col: float, delta_row: float,
    pixel_scale_arcsec: float, rotation_deg: float, flip_parity: bool = False,
) -> tuple[float, float]:
    """Updates a previously-fitted polar-alignment axis (prev_axis_ra_deg,
    prev_axis_dec_deg -- am5.gui.panels.AlignmentPanel's own 3-point
    fit_rotation_axis result) given how much the star field has visibly
    shifted (delta_col, delta_row pixels, this project's own image
    convention -- see project_radec_to_pixel's own docstring) between a
    reference frame (the fit's own last solved frame) and a later live
    frame, WITHOUT a fresh plate solve. Lets AlignmentPanel show a live-
    updating alt/az correction estimate while the operator turns the
    mount's altitude/azimuth adjusters, refreshed by cheap image
    correlation (camera/guiding.py's measure_frame_shift) instead of a
    real (multi-second) astrometry.net solve on every tick.

    Physical reasoning (rigid body): turning the alt/az adjusters
    physically rotates the WHOLE mount assembly -- RA axis, DEC axis,
    OTA, and camera -- together, by some rotation R. Since the mechanical
    axis is rigidly part of that same assembly, its own sky position
    moves by exactly that same R. But because the CAMERA also rotates by
    R, a star that hasn't actually moved appears, in the new frame, to
    have shifted by -R relative to where it was in the old frame (turning
    the camera towards a star makes it move toward frame centre, i.e. the
    apparent shift is the negation of the camera's own real rotation). So
    the axis's own new position is the OLD axis position rotated by the
    NEGATION of the star field's own observed apparent shift, not by the
    shift itself -- getting this sign backwards would show the operator
    an estimate correcting exactly the wrong way, the same class of bug
    already found and fixed once this session for the (unrelated)
    correction-arrow overlay.

    delta_col/delta_row -> (east_arcsec, north_arcsec) is the algebraic
    inverse of project_radec_to_pixel's own (east_arcsec, north_arcsec)
    -> (delta_col, delta_row) linear step (rotation-then-scale by a
    matrix that is its own inverse up to the scale factor, since
    [[cos,sin],[sin,-cos]] squares to the identity) -- so this reuses
    that function's exact same trig, just solved for the other pair of
    unknowns, rather than a separately-derived formula. flip_parity must
    be undone in the same order project_radec_to_pixel applies it
    (negate north AFTER recovering it from the linear inverse, mirroring
    that function negating it BEFORE the same linear step)."""
    rot = math.radians(rotation_deg)
    east_arcsec = pixel_scale_arcsec * (delta_col * math.cos(rot) + delta_row * math.sin(rot))
    north_arcsec = pixel_scale_arcsec * (delta_col * math.sin(rot) - delta_row * math.cos(rot))
    if flip_parity:
        north_arcsec = -north_arcsec
    return offset_radec_by_east_north(prev_axis_ra_deg, prev_axis_dec_deg, -east_arcsec, -north_arcsec)
