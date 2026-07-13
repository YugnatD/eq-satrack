import math
from datetime import datetime, timezone

import pytest

from am5.angles import angular_separation_deg, equatorial_to_altaz, gmst_deg


def test_angular_separation_deg_same_point_is_zero():
    assert angular_separation_deg(120.0, 45.0, 120.0, 45.0) == pytest.approx(0.0, abs=1e-9)


def test_angular_separation_deg_antipodal_points_is_180():
    assert angular_separation_deg(0.0, 0.0, 180.0, 0.0) == pytest.approx(180.0, abs=1e-6)
    assert angular_separation_deg(0.0, 90.0, 0.0, -90.0) == pytest.approx(180.0, abs=1e-6)


def test_angular_separation_deg_matches_small_angle_tangent_plane_approximation():
    # For a small separation, the great-circle distance should agree with
    # the tangent-plane hypot(d_ra*cos(dec), d_dec) approximation that
    # jog_goto's divergence guard used to (wrongly) rely on for ANY
    # separation -- confirming this isn't just "a different formula" but
    # the correct generalization of it.
    ra1, dec1 = 100.0, 30.0
    d_ra, d_dec = 0.02, 0.015  # a few dozen arcsec -- well within the small-angle regime
    ra2, dec2 = ra1 + d_ra, dec1 + d_dec
    tangent_plane_deg = math.hypot(d_ra * math.cos(math.radians(dec1)), d_dec)
    assert angular_separation_deg(ra1, dec1, ra2, dec2) == pytest.approx(tangent_plane_deg, rel=1e-3)


def test_angular_separation_deg_is_symmetric():
    a = angular_separation_deg(10.0, 20.0, 200.0, -40.0)
    b = angular_separation_deg(200.0, -40.0, 10.0, 20.0)
    assert a == pytest.approx(b)


def test_angular_separation_deg_stays_monotonic_where_the_tangent_plane_formula_does_not():
    # Real incident this session: a jog_goto from RA~13h DEC~67deg to
    # Deneb (RA~20.69h DEC~45.28deg, ~57 deg away) saw its divergence
    # guard's error metric GROW even while both raw RA and DEC
    # differences were individually shrinking, because the tangent-plane
    # formula's cos(dec) term grows as dec drops away from the pole --
    # meaningless outside the small-angle regime it's only valid for.
    # angular_separation_deg must not reproduce that: as the mount here
    # moves in a straight line towards the target (in real RA/DEC steps,
    # like the real incident), separation should shrink at every step,
    # not just at the endpoints.
    target_ra_deg, target_dec_deg = 310.357973, 45.280334  # Deneb
    start_ra_deg, start_dec_deg = 196.5, 70.59  # ~13.1h, near the pole
    steps = 20
    seps = [
        angular_separation_deg(
            start_ra_deg + (target_ra_deg - start_ra_deg) * i / steps,
            start_dec_deg + (target_dec_deg - start_dec_deg) * i / steps,
            target_ra_deg, target_dec_deg,
        )
        for i in range(steps + 1)
    ]
    assert seps[-1] == pytest.approx(0.0, abs=1e-6)
    assert all(a >= b - 1e-9 for a, b in zip(seps, seps[1:])), seps


def test_gmst_deg_is_in_range():
    when = datetime(2026, 7, 10, 17, 28, 59, tzinfo=timezone.utc)
    assert 0.0 <= gmst_deg(when) < 360.0


def test_equatorial_to_altaz_zenith_target_is_alt_90():
    when = datetime.now(timezone.utc)
    lat_deg, lon_deg = 46.18, 6.14
    lst_deg = (gmst_deg(when) + lon_deg) % 360.0
    az, alt = equatorial_to_altaz(ra_deg=lst_deg, dec_deg=lat_deg, lat_deg=lat_deg, lon_deg=lon_deg, when=when)
    assert alt == pytest.approx(90.0, abs=1e-4)


def test_equatorial_to_altaz_target_90deg_away_in_ra_is_near_horizon():
    when = datetime.now(timezone.utc)
    lat_deg, lon_deg = 46.18, 6.14
    lst_deg = (gmst_deg(when) + lon_deg) % 360.0
    _, alt = equatorial_to_altaz(ra_deg=(lst_deg + 90.0) % 360.0, dec_deg=0.0, lat_deg=lat_deg, lon_deg=lon_deg, when=when)
    assert alt == pytest.approx(0.0, abs=1e-3)
