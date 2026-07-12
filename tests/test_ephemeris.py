import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from skyfield.api import EarthSatellite, load, wgs84

from am5.ephemeris import compute_trajectory, find_next_pass, find_passes, load_iss_tle, load_satellite_tle, meridian_crossings

# Fixed, well-formed TLE used only to keep these tests deterministic and
# network-free — not claimed to reflect the ISS's current orbit.
TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9006"
TLE_LINE2 = "2 25544  51.6400 208.9163 0006317  69.9862 25.2825 15.49560500000000"

GENEVA = wgs84.latlon(46.18, 6.14)


@pytest.fixture
def satellite() -> EarthSatellite:
    ts = load.timescale()
    return EarthSatellite(TLE_LINE1, TLE_LINE2, "ISS (fixture)", ts)


def test_load_iss_tle_uses_cache_without_network(tmp_path):
    cache_path = tmp_path / "iss.tle"
    cache_path.write_text(f"ISS (ZARYA)\n{TLE_LINE1}\n{TLE_LINE2}\n")
    sat = load_iss_tle(cache_path, max_age_hours=1e9)  # never stale -> must not touch the network
    assert sat.name.startswith("ISS")


def test_load_satellite_tle_uses_cache_without_network_for_any_catnr(tmp_path):
    # Same cache-hit path as load_iss_tle, but for an arbitrary satellite
    # (see PassesPanel's target picker in am5/gui/panels.py) -- the catalog
    # number only matters for which URL a cache *miss* would fetch.
    cache_path = tmp_path / "tle_48274.tle"
    cache_path.write_text(f"CSS (TIANHE)\n{TLE_LINE1}\n{TLE_LINE2}\n")
    sat = load_satellite_tle(48274, cache_path, max_age_hours=1e9)
    assert sat.name.startswith("CSS")


def test_find_passes_magnitude_ref_none_gives_nan(satellite):
    # PassesPanel passes magnitude_ref=None for any satellite besides the
    # ISS, since estimate_iss_magnitude's distance scaling is calibrated
    # from a real ISS capture and meaningless for another object's actual
    # size/reflectivity -- see find_passes' docstring.
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    passes = find_passes(satellite, GENEVA, t0=t0, horizon_deg=10.0, lookahead_hours=48.0, magnitude_ref=None)
    assert passes
    assert all(math.isnan(p.magnitude_estimate) for p in passes)


def test_find_next_pass_ordering(satellite):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = find_next_pass(satellite, GENEVA, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)
    assert window.t_rise < window.t_culminate < window.t_set
    assert 10.0 <= window.max_elevation_deg <= 90.0
    assert 300.0 < window.distance_km < 3000.0  # sane ISS slant-range bounds
    assert -5.0 < window.magnitude_estimate < 2.0  # sane ISS brightness bounds


def test_find_next_pass_raises_when_none_found(satellite):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        find_next_pass(satellite, GENEVA, t0=t0, horizon_deg=89.9, lookahead_hours=1.0)


def test_find_passes_returns_multiple_in_order(satellite):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    passes = find_passes(satellite, GENEVA, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)
    assert len(passes) >= 2  # a 48h window at 10deg horizon should show several ISS passes
    for a, b in zip(passes, passes[1:]):
        assert a.t_set < b.t_rise
    assert passes[0] == find_next_pass(satellite, GENEVA, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)


def test_find_passes_empty_list_when_none_found(satellite):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert find_passes(satellite, GENEVA, t0=t0, horizon_deg=89.9, lookahead_hours=1.0) == []


def test_compute_trajectory_shapes_and_physical_bounds(satellite):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = find_next_pass(satellite, GENEVA, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)
    traj = compute_trajectory(satellite, GENEVA, window.t_rise, window.t_set, step_s=0.2)

    n = len(traj.t_unix)
    assert n > 10
    for arr in (traj.ra_deg, traj.dec_deg, traj.dra_dt_deg_s, traj.ddec_dt_deg_s, traj.alt_deg, traj.az_deg, traj.ha_hours):
        assert len(arr) == n
        assert np.all(np.isfinite(arr))

    assert np.all(traj.dec_deg >= -90.0) and np.all(traj.dec_deg <= 90.0)
    # on-sky angular speed, not raw dRA/dt (RA degrees near the pole move fast on the axis
    # but cover little real sky distance — see the brief's own zenith-speed derivation)
    sky_speed = np.sqrt((traj.dra_dt_deg_s * np.cos(np.radians(traj.dec_deg))) ** 2 + traj.ddec_dt_deg_s ** 2)
    assert np.all(sky_speed < 2.0)  # ISS never exceeds ~1 deg/s even at zenith


def test_interpolate_holds_zero_rate_outside_window(satellite):
    # An early/late tracking-loop query must hold position (not drift at
    # the boundary's real, usually large, angular rate) -- see
    # Trajectory.interpolate's docstring for the incident this fixes.
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = find_next_pass(satellite, GENEVA, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)
    traj = compute_trajectory(satellite, GENEVA, window.t_rise, window.t_set, step_s=0.2)

    ra_before, dec_before, dra_before, ddec_before = traj.interpolate(traj.t_unix[0] - 3600.0)
    assert dra_before == 0.0 and ddec_before == 0.0
    assert ra_before == pytest.approx(traj.ra_deg[0])
    assert dec_before == pytest.approx(traj.dec_deg[0])

    ra_after, dec_after, dra_after, ddec_after = traj.interpolate(traj.t_unix[-1] + 3600.0)
    assert dra_after == 0.0 and ddec_after == 0.0
    assert ra_after == pytest.approx(traj.ra_deg[-1])
    assert dec_after == pytest.approx(traj.dec_deg[-1])

    # inside the window, rates must still be the real (generally nonzero) values
    _, _, dra_inside, ddec_inside = traj.interpolate(traj.t_unix[len(traj.t_unix) // 2])
    assert abs(dra_inside) > 0.0 or abs(ddec_inside) > 0.0


def test_compute_trajectory_rejects_backwards_window(satellite):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        compute_trajectory(satellite, GENEVA, t0, t0 - timedelta(seconds=1))


def test_meridian_crossings_match_ha_sign_changes(satellite):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = find_next_pass(satellite, GENEVA, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)
    traj = compute_trajectory(satellite, GENEVA, window.t_rise, window.t_set, step_s=0.2)
    crossings = meridian_crossings(traj)
    # every crossing must fall strictly within the trajectory's own time span
    for t in crossings:
        assert traj.t_unix[0] <= t.timestamp() <= traj.t_unix[-1]
    # and the number of crossings must match the number of sign changes actually
    # present in ha_hours (self-consistency, not a claim about the real ISS)
    expected = int(np.count_nonzero(np.diff(np.sign(traj.ha_hours)) != 0))
    assert len(crossings) == expected
