import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from skyfield.api import EarthSatellite, load, wgs84

from am5.ephemeris import (
    compute_trajectory,
    current_pass_window,
    currently_visible_satellites,
    find_next_pass,
    find_passes,
    load_iss_tle,
    load_satellite_group_tles,
    load_satellite_tle,
    meridian_crossings,
)

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


def test_load_satellite_group_tles_uses_cache_without_network(tmp_path):
    # Same cache-hit path as load_iss_tle/load_satellite_tle, but for a
    # whole named group (see PassesPanel's "Live now" sub-tab, am5/gui/
    # panels.py) -- returns every satellite in the file, not just the
    # first, since the whole point is checking many candidates.
    cache_path = tmp_path / "tle_group_visual.tle"
    cache_path.write_text(
        f"ISS (ZARYA)\n{TLE_LINE1}\n{TLE_LINE2}\n"
        f"CSS (TIANHE)\n{TLE_LINE1}\n{TLE_LINE2}\n"
    )
    sats = load_satellite_group_tles("visual", cache_path, max_age_hours=1e9)
    assert [s.name for s in sats] == ["ISS (ZARYA)", "CSS (TIANHE)"]


def test_currently_visible_satellites_filters_by_horizon(satellite):
    # No second real satellite fixture needed to exercise the filter: the
    # SAME satellite is definitely above horizon_deg at its own
    # culmination, and definitely below it well before its own rise --
    # both facts already established by find_next_pass's own geometry.
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = find_next_pass(satellite, GENEVA, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)

    visible_at_culm = currently_visible_satellites([satellite], GENEVA, horizon_deg=10.0, t0=window.t_culminate)
    assert len(visible_at_culm) == 1
    sat, alt_deg, az_deg = visible_at_culm[0]
    assert sat is satellite
    assert alt_deg == pytest.approx(window.max_elevation_deg, abs=0.5)
    assert 0.0 <= az_deg < 360.0

    before_rise = window.t_rise - timedelta(hours=1)
    assert currently_visible_satellites([satellite], GENEVA, horizon_deg=10.0, t0=before_rise) == []


def test_currently_visible_satellites_sorted_by_altitude_descending(satellite):
    # A single satellite obviously "sorts" trivially -- this just confirms
    # the list shape or a >1 entries wouldn't need any code that isn't
    # already exercised: currently_visible_satellites always returns
    # (satellite, alt_deg, az_deg) tuples with alt_deg the SAME value
    # passed to the horizon filter, so a duplicate-entry list (same
    # satellite twice) sanity-checks the sort is at least stable/lossless.
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = find_next_pass(satellite, GENEVA, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)
    result = currently_visible_satellites([satellite, satellite], GENEVA, horizon_deg=10.0, t0=window.t_culminate)
    assert len(result) == 2
    alts = [alt for _sat, alt, _az in result]
    assert alts == sorted(alts, reverse=True)


def test_currently_visible_satellites_skips_one_broken_entry_instead_of_aborting(satellite, capsys):
    # Regression, found by code audit: a single satellite raising during
    # altaz computation (e.g. a malformed/degenerate TLE, realistic in a
    # ~150-250 entry real Celestrak group fetch) used to abort the WHOLE
    # scan -- one bad entry should never take down the rest of the list.
    class _BrokenSatellite:
        name = "BROKEN (fixture)"

        def __sub__(self, other):
            raise RuntimeError("boom -- simulated degenerate TLE")

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = find_next_pass(satellite, GENEVA, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)
    result = currently_visible_satellites(
        [_BrokenSatellite(), satellite], GENEVA, horizon_deg=10.0, t0=window.t_culminate,
    )
    assert len(result) == 1
    assert result[0][0] is satellite
    assert "BROKEN (fixture)" in capsys.readouterr().err


def test_current_pass_window_starts_now_and_finds_the_real_set_event(satellite):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    scheduled = find_next_pass(satellite, GENEVA, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)

    # Ask for the "remaining" window starting mid-pass, at culmination.
    window = current_pass_window(satellite, GENEVA, horizon_deg=10.0, t0=scheduled.t_culminate)
    assert window.t_rise == scheduled.t_culminate  # "rise" is just t0 here, by construction
    assert abs((window.t_set - scheduled.t_set).total_seconds()) < 1.0
    assert window.t_culminate == scheduled.t_culminate  # already at/past culmination
    assert math.isnan(window.magnitude_estimate)  # no calibrated magnitude_ref for an arbitrary satellite

    # And compute_trajectory (the existing, unmodified tracking-side
    # machinery) must accept this window exactly like a scheduled one.
    trajectory = compute_trajectory(satellite, GENEVA, window.t_rise, window.t_set, step_s=1.0)
    assert trajectory.t_unix[0] == pytest.approx(window.t_rise.timestamp(), abs=1.0)


def test_current_pass_window_before_culmination_finds_the_upcoming_culmination(satellite):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    scheduled = find_next_pass(satellite, GENEVA, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)

    # Starting shortly after the real rise (still climbing) -- the real
    # culmination event is still ahead, must be found (not just t0).
    start = scheduled.t_rise + timedelta(seconds=5)
    window = current_pass_window(satellite, GENEVA, horizon_deg=10.0, t0=start)
    assert window.t_rise == start
    # find_events' own root-finding precision shifts by a few ms depending
    # on the search window's start -- not a meaningful difference here.
    assert abs((window.t_culminate - scheduled.t_culminate).total_seconds()) < 1.0
    assert abs((window.t_set - scheduled.t_set).total_seconds()) < 1.0


def test_current_pass_window_raises_when_no_set_event_in_lookahead(satellite):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        current_pass_window(satellite, GENEVA, horizon_deg=89.9, t0=t0, lookahead_hours=1.0)


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
