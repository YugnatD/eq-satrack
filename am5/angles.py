"""Angle wraparound helpers shared by the mount-characterization analysis
and the ephemeris trajectory math — RA/azimuth both wrap, and getting the
wrap wrong silently corrupts a velocity fit (see characterize.py's history)."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np


def unwrap_deg(values: np.ndarray) -> np.ndarray:
    """Unwrap a degrees series with 360 deg period (RA/azimuth crossing 0/360)."""
    return np.degrees(np.unwrap(np.radians(values)))


def circular_diff_deg(a: float, b: float) -> float:
    """Signed minimal difference a-b for a 360-degree-periodic quantity, in [-180, 180)."""
    return ((a - b + 180.0) % 360.0) - 180.0


def circular_diff_hours(a: float, b: float) -> float:
    """Signed minimal difference a-b for a 24-hour-periodic quantity (RA), in [-12, 12)."""
    return ((a - b + 12.0) % 24.0) - 12.0


def angular_separation_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    """Great-circle angular separation between two RA/DEC points, in
    degrees, via the haversine formula -- correct for any separation.

    Not the same as hypot(d_ra*cos(dec), d_dec): that's a tangent-plane
    (small-angle) approximation that only holds for separations of a few
    degrees. Confirmed on real hardware to actively mislead beyond that --
    a jog_goto's divergence guard using the tangent-plane formula for a
    ~100+ deg initial separation (a GOTO to an arbitrary star, not the
    short final-approach jog_goto is meant for) reported an INCREASING
    error even while both raw RA and DEC differences were individually
    shrinking, because cos(dec) grows as dec moves away from the pole --
    tripping a false "diverged" abort with correct calibration and no
    pier flip involved."""
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1_deg, dec1_deg, ra2_deg, dec2_deg))
    sin_half_dec = math.sin((dec2 - dec1) / 2.0)
    sin_half_ra = math.sin((ra2 - ra1) / 2.0)
    a = sin_half_dec**2 + math.cos(dec1) * math.cos(dec2) * sin_half_ra**2
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(a))))


def gmst_deg(when: datetime) -> float:
    """Greenwich Mean Sidereal Time, in degrees, good to a few arcsec --
    plenty for anything in this project that isn't sent over the wire to
    real hardware (which gets its own site/time via
    Mount.sync_site_and_time()). Standard IAU-82-style polynomial in UT1
    Julian centuries from J2000, treating UTC as UT1 (off by <1s,
    irrelevant here). Verified against a real zenith-transit sanity check
    (see am5/mock_mount.py's history)."""
    jd = 2440587.5 + when.timestamp() / 86400.0
    d = jd - 2451545.0
    t = d / 36525.0
    gmst = 280.46061837 + 360.98564736629 * d + 0.000387933 * t * t - t * t * t / 38710000.0
    return gmst % 360.0


def equatorial_to_altaz(ra_deg: float, dec_deg: float, lat_deg: float, lon_deg: float, when: datetime) -> tuple[float, float]:
    """(az_deg, alt_deg) via direct spherical trig from RA/DEC (degrees) +
    site + time -- no ephemeris/ephemeris-file needed. Fine for stars
    (parallax/light-time from Earth's surface is negligible) and for a
    satellite's already-topocentric RA/DEC (computed upstream via
    Skyfield, see am5/ephemeris.py)."""
    lst_deg = (gmst_deg(when) + lon_deg) % 360.0
    ha_deg = (lst_deg - ra_deg) % 360.0
    lat = math.radians(lat_deg)
    dec = math.radians(dec_deg)
    ha = math.radians(ha_deg)
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
    alt = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))
    cos_alt = math.cos(math.radians(alt))
    if abs(cos_alt) < 1e-9:
        az = 0.0
    else:
        cos_az = (math.sin(dec) - math.sin(lat) * math.sin(math.radians(alt))) / (math.cos(lat) * cos_alt)
        az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
        if math.sin(ha) > 0:
            az = 360.0 - az
    return az, alt


def equatorial_series_to_altaz(
    ra_deg: np.ndarray, dec_deg: np.ndarray, lat_deg: float, lon_deg: float, when: datetime
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equatorial_to_altaz for a whole RA/DEC series at a single
    reference time -- e.g. to redraw a planned trajectory as it would
    appear "if it were happening right now" (see am5/gui/panels.py's
    SkyMapWidget, used to align a rehearsal GOTO/simulated track with the
    live telescope marker, both evaluated at the same instant)."""
    lst_deg = (gmst_deg(when) + lon_deg) % 360.0
    ha_deg = (lst_deg - ra_deg) % 360.0
    lat = math.radians(lat_deg)
    dec = np.radians(dec_deg)
    ha = np.radians(ha_deg)
    sin_alt = math.sin(lat) * np.sin(dec) + math.cos(lat) * np.cos(dec) * np.cos(ha)
    alt = np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))
    cos_alt = np.cos(np.radians(alt))
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_az = (np.sin(dec) - math.sin(lat) * np.sin(np.radians(alt))) / (math.cos(lat) * cos_alt)
    az = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    az = np.where(np.sin(ha) > 0, 360.0 - az, az)
    az = np.where(np.abs(cos_alt) < 1e-9, 0.0, az)
    return az, alt
