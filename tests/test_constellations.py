from datetime import datetime, timezone

import pytest
from skyfield.api import wgs84

from am5.constellations import CONSTELLATIONS, constellations_altaz


def test_constellations_have_valid_line_indices():
    for shape in CONSTELLATIONS:
        n = len(shape.stars_radec)
        assert n >= 3
        for i, j in shape.lines:
            assert 0 <= i < n
            assert 0 <= j < n


def test_constellations_altaz_matches_shape_count_and_order():
    site = wgs84.latlon(46.18, 6.14)
    when = datetime(2026, 1, 15, 20, 0, 0, tzinfo=timezone.utc)
    results = constellations_altaz(site, when)
    assert len(results) == len(CONSTELLATIONS)
    for shape, result in zip(CONSTELLATIONS, results):
        assert result.name == shape.name
        assert result.lines == shape.lines
        assert len(result.stars_azalt) == len(shape.stars_radec)
        for az, alt in result.stars_azalt:
            assert 0.0 <= az < 360.0
            assert -90.0 <= alt <= 90.0


def test_polaris_altitude_matches_observer_latitude():
    # Polaris sits within ~1 deg of the north celestial pole, so its
    # altitude from a given latitude should closely match that latitude.
    site = wgs84.latlon(46.18, 6.14)
    when = datetime(2026, 1, 15, 20, 0, 0, tzinfo=timezone.utc)
    results = constellations_altaz(site, when)
    ursa_minor = next(r for r in results if r.name == "Ursa Minor")
    polaris_az, polaris_alt = ursa_minor.stars_azalt[0]
    assert polaris_alt == pytest.approx(46.18, abs=1.0)
