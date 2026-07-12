from datetime import datetime, timezone

import pytest

from am5.angles import equatorial_to_altaz, gmst_deg


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
