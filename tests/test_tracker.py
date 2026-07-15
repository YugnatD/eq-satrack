import csv
import io
import math
import threading
import time
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest

from am5.angles import angular_separation_deg, gmst_deg
from am5.constants import SIDEREAL_DEG_PER_S
from am5.ephemeris import Trajectory
from am5.mock_mount import MockConfig, MockMount
from am5.mount import Mount
from am5.safety import SafetyGuard
from am5.tracker import (
    TRACKING_CSV_FIELDS,
    AxisSigns,
    LiveOffsets,
    MountLagResult,
    TrackingConfig,
    TrackingRunaway,
    calibrate_directions,
    decompose_error,
    measure_mount_lag,
    _along_cross_rate_to_equatorial,
    _perp_rate_components,
    _pick_direction,
    run_tracking_loop,
)


def test_pick_direction_matches_convention():
    assert _pick_direction(5.0, sign_convention=1.0, positive_dir="e", negative_dir="w") == "e"
    assert _pick_direction(-5.0, sign_convention=1.0, positive_dir="e", negative_dir="w") == "w"
    # flipped wiring: a positive commanded rate now needs the "negative" direction command
    assert _pick_direction(5.0, sign_convention=-1.0, positive_dir="e", negative_dir="w") == "w"


def test_perp_rate_components_are_perpendicular_to_velocity():
    # velocity purely along RA (tangent-plane) -> perpendicular nudge should be pure DEC
    dra_extra, ddec_extra = _perp_rate_components(dec_deg=0.0, dra_dt_deg_s=1.0, ddec_dt_deg_s=0.0, sign=1.0)
    assert dra_extra == pytest.approx(0.0, abs=1e-9)
    assert ddec_extra != 0.0

    # opposite sign flips the nudge direction
    dra2, ddec2 = _perp_rate_components(dec_deg=0.0, dra_dt_deg_s=1.0, ddec_dt_deg_s=0.0, sign=-1.0)
    assert ddec2 == pytest.approx(-ddec_extra)


def test_perp_rate_components_zero_sign_is_noop():
    assert _perp_rate_components(45.0, 1.0, 1.0, sign=0.0) == (0.0, 0.0)


def test_axis_signs_update_pier_side_first_call_just_records():
    # No prior reading to compare against -- must not flip dec on the
    # very first observation, only start tracking which side is current.
    signs = AxisSigns(ra=1.0, dec=1.0)
    flipped = signs.update_pier_side("E")
    assert flipped is False
    assert signs.dec == 1.0
    assert signs.calibrated_pier_side == "E"


def test_axis_signs_update_pier_side_flips_dec_on_change():
    # Confirmed on real AM3 hardware: calibrating on pier side E gave
    # dec=+1, immediately re-calibrating on side W (same session, only the
    # pier side changed via a GOTO) gave dec=-1, ra unchanged.
    signs = AxisSigns(ra=1.0, dec=1.0, calibrated_pier_side="E")
    flipped = signs.update_pier_side("W")
    assert flipped is True
    assert signs.dec == -1.0
    assert signs.ra == 1.0  # unaffected -- RA's axis isn't rotated by a pier flip
    assert signs.calibrated_pier_side == "W"


def test_axis_signs_update_pier_side_noop_when_unchanged():
    signs = AxisSigns(ra=1.0, dec=-1.0, calibrated_pier_side="W")
    flipped = signs.update_pier_side("W")
    assert flipped is False
    assert signs.dec == -1.0


def test_axis_signs_update_pier_side_ignores_home_and_unknown():
    # 'N' (home/zero position) and None/unknown don't tell us which side
    # we'd actually be tracking from -- recording them would risk a false
    # "flip" the next time a real E/W reading comes in.
    signs = AxisSigns(ra=1.0, dec=1.0, calibrated_pier_side="E")
    assert signs.update_pier_side("N") is False
    assert signs.update_pier_side(None) is False
    assert signs.calibrated_pier_side == "E"
    assert signs.dec == 1.0


def test_axis_signs_update_pier_side_can_flip_back_and_forth():
    signs = AxisSigns(ra=1.0, dec=1.0, calibrated_pier_side="E")
    assert signs.update_pier_side("W") is True
    assert signs.dec == -1.0
    assert signs.update_pier_side("E") is True
    assert signs.dec == 1.0


def test_decompose_error_pure_along_track():
    # error parallel to velocity direction -> all along-track, no cross-track
    along, cross = decompose_error(d_ra_deg=1.0, d_dec_deg=0.0, dec_deg=0.0, dra_dt_deg_s=1.0, ddec_dt_deg_s=0.0)
    assert along == pytest.approx(1.0)
    assert cross == pytest.approx(0.0, abs=1e-9)


def test_decompose_error_pure_cross_track():
    along, cross = decompose_error(d_ra_deg=0.0, d_dec_deg=1.0, dec_deg=0.0, dra_dt_deg_s=1.0, ddec_dt_deg_s=0.0)
    assert along == pytest.approx(0.0, abs=1e-9)
    assert abs(cross) == pytest.approx(1.0)


def test_along_cross_rate_to_equatorial_is_the_inverse_of_decompose_error():
    # Round-trip: project a raw (dra_dt, ddec_dt) "error" onto (along,
    # cross) via decompose_error, then back via the new helper -- should
    # recover the original vector (up to the max_deg_s clamp, set high
    # enough here to not engage).
    dec_deg, dra_dt, ddec_dt = 20.0, 0.003, 0.001
    for raw_dra, raw_ddec in [(0.002, -0.0015), (-0.001, 0.0025), (0.0, 0.001)]:
        along, cross = decompose_error(raw_dra, raw_ddec, dec_deg, dra_dt, ddec_dt)
        back_dra, back_ddec = _along_cross_rate_to_equatorial(along, cross, dec_deg, dra_dt, ddec_dt, max_deg_s=10.0)
        assert back_dra == pytest.approx(raw_dra, abs=1e-9)
        assert back_ddec == pytest.approx(raw_ddec, abs=1e-9)


def test_along_cross_rate_to_equatorial_clamps_combined_magnitude():
    dec_deg, dra_dt, ddec_dt = 0.0, 1.0, 0.0
    dra, ddec = _along_cross_rate_to_equatorial(
        along_rate_deg_s=1.0, cross_rate_deg_s=1.0, dec_deg=dec_deg,
        dra_dt_deg_s=dra_dt, ddec_dt_deg_s=ddec_dt, max_deg_s=0.5,
    )
    assert math.hypot(dra * math.cos(math.radians(dec_deg)), ddec) == pytest.approx(0.5, abs=1e-9)


def test_along_cross_rate_to_equatorial_handles_zero_velocity():
    # No track direction to project onto (e.g. mount at rest) -- must not
    # divide by zero, and should still produce something bounded.
    dra, ddec = _along_cross_rate_to_equatorial(0.01, 0.01, 45.0, 0.0, 0.0, max_deg_s=0.02)
    assert math.isfinite(dra)
    assert math.isfinite(ddec)


def test_live_offsets_perp_pulse_expires():
    offsets = LiveOffsets()
    offsets.trigger_perp_pulse(sign=1.0, duration_s=0.05)
    _, perp = offsets.snapshot()
    assert perp == 1.0
    time.sleep(0.08)
    _, perp = offsets.snapshot()
    assert perp == 0.0


def test_calibrate_directions_records_the_pier_side_it_calibrated_on():
    mock = MockMount(MockConfig(rv_mode="per_axis"))
    mount = Mount(mock)
    try:
        signs = calibrate_directions(mount)
        assert signs.calibrated_pier_side == mount.get_pier_side()
        assert signs.calibrated_pier_side in ("E", "W")  # not at home -- see MockConfig.at_home default
    finally:
        mount.stop()
        mount.close()


def test_calibrate_directions_normalizes_home_pier_side_to_none():
    # Regression: if the mount is still (or reports as) at home ('N', no
    # determinate side) right when calibration checks pier side, storing
    # 'N' as-is would make the *next* real E/W reading look like a false
    # "flip" to update_pier_side() -- 'N' != 'E' is True even though
    # nothing physically flipped, since there was no real side to flip
    # from. None matches update_pier_side's own "first reading just
    # records" handling instead.
    mock = MockMount(MockConfig(rv_mode="per_axis"))
    mount = Mount(mock)
    try:
        with patch.object(Mount, "get_pier_side", return_value="N"):
            signs = calibrate_directions(mount)
        assert signs.calibrated_pier_side is None
        assert signs.update_pier_side("E") is False  # first real reading just records, no false flip
        assert signs.calibrated_pier_side == "E"
    finally:
        mount.stop()
        mount.close()


def test_measure_mount_lag_recovers_the_mocks_known_ramp_constant():
    # MockMount simulates a real first-order mechanical ramp (tau_s=0.15,
    # see mock_mount.py) -- a 1st-order step response reaches
    # settle_fraction=0.9 of steady-state at t = -tau*ln(1-0.9) =~
    # 2.303*tau =~ 0.345s. This is the one real ground truth available
    # without hardware, so it's what the algorithm is checked against.
    # Seeded: the mock's simulated serial latency is randomized per-call
    # (see MockMount._sample_latency), which is real enough noise on a
    # measurement that's inherently timing-sensitive that an unseeded run
    # swings several tens of ms -- seeding makes the assertion meaningful
    # instead of flaky.
    mock = MockMount(MockConfig(rv_mode="per_axis"), seed=1)
    mount = Mount(mock)
    try:
        result = measure_mount_lag(mount, rate_x=100.0, duration_s=1.0)
    finally:
        mount.stop()
        mount.close()

    assert isinstance(result, MountLagResult)
    assert result.lag_s == pytest.approx(0.345, abs=0.1)
    expected_rate_arcsec_s = 100.0 * SIDEREAL_DEG_PER_S * 3600.0
    assert result.steady_rate_arcsec_s == pytest.approx(expected_rate_arcsec_s, rel=0.1)
    assert result.samples > 4


def test_measure_mount_lag_stops_the_manual_jog_when_done():
    # mount.stop("e") ends the manual jog issued during the measurement --
    # sidereal tracking (on by default, like real hardware) continues on
    # its own afterwards, so this only checks the rate drops back down
    # near sidereal instead of staying pinned at the ~1500"/s commanded
    # during the test -- not an exact match, real mock axis-state
    # transition timing isn't the thing under test here.
    mock = MockMount(MockConfig(rv_mode="per_axis"), seed=1)
    mount = Mount(mock)
    try:
        measure_mount_lag(mount, rate_x=100.0, duration_s=0.5)
        time.sleep(0.3)  # let the mock's own ramp settle back down
        radec_a = mount.get_radec()
        time.sleep(0.5)
        radec_b = mount.get_radec()
    finally:
        mount.stop()
        mount.close()
    residual_rate_arcsec_s = abs(radec_b.ra_hours - radec_a.ra_hours) * 15.0 * 3600.0 / 0.5
    assert residual_rate_arcsec_s < 100.0  # well below the ~1500"/s commanded during the test


def test_measure_mount_lag_aborts_early_when_signalled():
    mock = MockMount(MockConfig(rv_mode="per_axis"))
    mount = Mount(mock)
    abort = threading.Event()
    abort.set()  # already set -- should bail after at most one sample
    try:
        result = measure_mount_lag(mount, rate_x=100.0, duration_s=5.0, abort=abort)
    finally:
        mount.stop()
        mount.close()
    assert result.samples <= 2


def _make_constant_rate_trajectory(rate_x_sidereal: float, duration_s: float, start_ra: float, start_dec: float) -> Trajectory:
    n = int(duration_s * 50)
    t0 = time.time()
    t_unix = t0 + np.linspace(0, duration_s, n)
    rate_deg_s = rate_x_sidereal * SIDEREAL_DEG_PER_S
    elapsed = t_unix - t0
    ra_deg = start_ra + rate_deg_s * elapsed
    dec_deg = start_dec + rate_deg_s * elapsed
    return Trajectory(
        t_unix=t_unix, ra_deg=ra_deg, dec_deg=dec_deg,
        dra_dt_deg_s=np.full(n, rate_deg_s), ddec_dt_deg_s=np.full(n, rate_deg_s),
        alt_deg=np.zeros(n), az_deg=np.zeros(n), ha_hours=np.zeros(n), distance_km=np.full(n, 500.0),
    )


def test_tracking_loop_follows_constant_rate_trajectory():
    start_ra, start_dec = 10.0, 45.0
    rate_x = 50.0
    duration_s = 2.5
    trajectory = _make_constant_rate_trajectory(rate_x, duration_s + 1.0, start_ra, start_dec)

    mock = MockMount(MockConfig(rv_mode="per_axis", tracking_adds=True, start_ra_deg=start_ra, start_dec_deg=start_dec))
    mount = Mount(mock)
    safety = SafetyGuard(mount, watchdog_timeout=5.0)
    fh = io.StringIO()
    writer = csv.DictWriter(fh, fieldnames=TRACKING_CSV_FIELDS)
    writer.writeheader()

    try:
        run_tracking_loop(
            mount, safety, trajectory, AxisSigns(ra=1.0, dec=1.0), LiveOffsets(), writer,
            duration_s=duration_s, config=TrackingConfig(loop_hz=20.0),
        )
        actual = mount.get_radec()
    finally:
        mount.stop()
        safety.shutdown()
        mount.close()

    expected_ra = start_ra + rate_x * SIDEREAL_DEG_PER_S * duration_s
    expected_dec = start_dec + rate_x * SIDEREAL_DEG_PER_S * duration_s
    ra_error_arcsec = abs(actual.ra_hours * 15.0 - expected_ra) * 3600
    dec_error_arcsec = abs(actual.dec_deg - expected_dec) * 3600

    # The mock's first-order velocity ramp (tau=0.15s, see mock_mount.py)
    # never fully closes a step from 0 to the trajectory's rate: a 1st-order
    # lag system tracking a velocity step settles at a fixed catch-up lag of
    # rate * tau =~ 50x * SIDEREAL_DEG_PER_S * 0.15s =~ 113 arcsec. This test
    # only needs to rule out gross regressions (wrong axis, exceptions,
    # runaway divergence), not match that analytic figure exactly.
    assert ra_error_arcsec < 300.0
    assert dec_error_arcsec < 300.0


def test_tracking_loop_respects_axis_sign_flip():
    """If the wiring convention is flipped, the loop must still converge —
    it should pick the opposite direction command, not just get the sign wrong."""
    start_ra, start_dec = 10.0, 45.0
    rate_x = 50.0
    duration_s = 2.0
    trajectory = _make_constant_rate_trajectory(rate_x, duration_s + 1.0, start_ra, start_dec)

    mock = MockMount(MockConfig(rv_mode="per_axis", tracking_adds=True, start_ra_deg=start_ra, start_dec_deg=start_dec))
    mount = Mount(mock)
    safety = SafetyGuard(mount, watchdog_timeout=5.0)
    fh = io.StringIO()
    writer = csv.DictWriter(fh, fieldnames=TRACKING_CSV_FIELDS)
    writer.writeheader()

    try:
        # sign_convention=-1 means _pick_direction should still land on the
        # command that produces a positive rate on this (relabeled) mock axis
        run_tracking_loop(
            mount, safety, trajectory, AxisSigns(ra=-1.0, dec=-1.0), LiveOffsets(), writer,
            duration_s=duration_s, config=TrackingConfig(loop_hz=20.0),
        )
        actual = mount.get_radec()
    finally:
        mount.stop()
        safety.shutdown()
        mount.close()

    # with a flipped convention but correct _pick_direction logic, RA should
    # have moved the WRONG way relative to the "e means +RA" mock, since we
    # deliberately told it the convention is flipped — confirms the sign
    # plumbing is actually being applied, not silently ignored.
    assert actual.ra_hours * 15.0 < start_ra


def test_tracking_loop_auto_stops_on_runaway():
    # Mount starts 15 deg away in DEC from where the trajectory says it
    # should be -- stands in for a wrong axis sign / bad model / stall that
    # leaves the mount diverging. The loop must trip the runaway guard and
    # stop, not keep commanding high-speed motion.
    start_ra = 10.0
    trajectory = _make_constant_rate_trajectory(50.0, 5.0, start_ra, start_dec=45.0)
    mock = MockMount(MockConfig(rv_mode="per_axis", tracking_adds=True, start_ra_deg=start_ra, start_dec_deg=30.0))
    mount = Mount(mock)
    safety = SafetyGuard(mount, watchdog_timeout=5.0)
    fh = io.StringIO()
    writer = csv.DictWriter(fh, fieldnames=TRACKING_CSV_FIELDS)
    writer.writeheader()

    try:
        with pytest.raises(TrackingRunaway):
            run_tracking_loop(
                mount, safety, trajectory, AxisSigns(ra=1.0, dec=1.0), LiveOffsets(), writer,
                duration_s=4.0, config=TrackingConfig(loop_hz=20.0, runaway_stop_deg=5.0),
            )
    finally:
        mount.stop()
        safety.shutdown()
        mount.close()


def test_tracking_loop_runaway_check_uses_great_circle_separation():
    # Regression for the same class of bug fixed in jog_goto's divergence
    # guard (see angular_separation_deg's docstring): the runaway check
    # used to compute error via a tangent-plane hypot(d_ra*cos(dec), d_dec)
    # approximation, only valid for small separations -- meaningless for
    # a real, large divergence, which is exactly when this guard matters
    # most (e.g. starting a tracking pass late without first slewing onto
    # target -- a real incident this session). total_error_deg isn't
    # itself exposed via on_tick, so confirm the wiring directly: the
    # loop must call angular_separation_deg (not reimplement the old
    # formula inline) to compute it.
    start_ra, start_dec = 196.5, 80.0
    trajectory = _make_constant_rate_trajectory(50.0, 2.0, start_ra, start_dec)
    mock = MockMount(MockConfig(rv_mode="per_axis", tracking_adds=True, start_ra_deg=start_ra, start_dec_deg=start_dec))
    mount = Mount(mock)
    safety = SafetyGuard(mount, watchdog_timeout=5.0)
    fh = io.StringIO()
    writer = csv.DictWriter(fh, fieldnames=TRACKING_CSV_FIELDS)
    writer.writeheader()
    try:
        with patch("am5.tracker.angular_separation_deg", wraps=angular_separation_deg) as spy:
            run_tracking_loop(
                mount, safety, trajectory, AxisSigns(ra=1.0, dec=1.0), LiveOffsets(), writer,
                duration_s=1.0, config=TrackingConfig(loop_hz=20.0, error_log_hz=20.0),
            )
    finally:
        mount.stop()
        safety.shutdown()
        mount.close()

    assert spy.call_count > 0


def test_tracking_loop_runaway_guard_disabled_with_zero():
    # runaway_stop_deg=0 disables the guard -- same 15deg offset must NOT raise.
    start_ra = 10.0
    trajectory = _make_constant_rate_trajectory(50.0, 3.0, start_ra, start_dec=45.0)
    mock = MockMount(MockConfig(rv_mode="per_axis", tracking_adds=True, start_ra_deg=start_ra, start_dec_deg=30.0))
    mount = Mount(mock)
    safety = SafetyGuard(mount, watchdog_timeout=5.0)
    fh = io.StringIO()
    writer = csv.DictWriter(fh, fieldnames=TRACKING_CSV_FIELDS)
    writer.writeheader()
    try:
        run_tracking_loop(  # must not raise
            mount, safety, trajectory, AxisSigns(ra=1.0, dec=1.0), LiveOffsets(), writer,
            duration_s=1.5, config=TrackingConfig(loop_hz=20.0, runaway_stop_deg=0.0),
        )
    finally:
        mount.stop()
        safety.shutdown()
        mount.close()


def test_tracking_loop_calls_on_tick():
    start_ra, start_dec = 10.0, 45.0
    trajectory = _make_constant_rate_trajectory(50.0, 2.0, start_ra, start_dec)
    mock = MockMount(MockConfig(rv_mode="per_axis", tracking_adds=True, start_ra_deg=start_ra, start_dec_deg=start_dec))
    mount = Mount(mock)
    safety = SafetyGuard(mount, watchdog_timeout=5.0)
    fh = io.StringIO()
    writer = csv.DictWriter(fh, fieldnames=TRACKING_CSV_FIELDS)
    writer.writeheader()

    ticks = []
    try:
        run_tracking_loop(
            mount, safety, trajectory, AxisSigns(ra=1.0, dec=1.0), LiveOffsets(), writer,
            duration_s=1.2, config=TrackingConfig(loop_hz=20.0, error_log_hz=5.0),
            on_tick=ticks.append,
        )
    finally:
        mount.stop()
        safety.shutdown()
        mount.close()

    assert len(ticks) >= 3
    for t in ticks:
        assert set(t) >= {"elapsed_s", "target_ra_deg", "target_dec_deg", "along_track_arcsec", "cross_track_arcsec"}


def test_tracking_loop_mount_lag_s_shifts_target_query_forward():
    # For a constant-rate trajectory, mount_lag_s doesn't change the
    # commanded velocity (it's the same everywhere) -- but it should shift
    # which point of the trajectory is used as the reported/logged target,
    # by exactly rate * mount_lag_s. This is a direct plumbing check on
    # t_query = now_wall + delta_t_s + mount_lag_s (see run_tracking_loop),
    # not a claim about real-hardware tracking improvement -- that depends
    # on real (non-constant) trajectory acceleration and is documented as
    # unvalidated on real hardware.
    start_ra, start_dec = 10.0, 45.0
    rate_x = 50.0
    duration_s = 1.5
    mount_lag_s = 0.3

    def run(lag_s):
        trajectory = _make_constant_rate_trajectory(rate_x, duration_s + 1.0, start_ra, start_dec)
        mock = MockMount(MockConfig(rv_mode="per_axis", tracking_adds=True, start_ra_deg=start_ra, start_dec_deg=start_dec))
        mount = Mount(mock)
        safety = SafetyGuard(mount, watchdog_timeout=5.0)
        fh = io.StringIO()
        writer = csv.DictWriter(fh, fieldnames=TRACKING_CSV_FIELDS)
        writer.writeheader()
        ticks = []
        try:
            run_tracking_loop(
                mount, safety, trajectory, AxisSigns(ra=1.0, dec=1.0), LiveOffsets(), writer,
                duration_s=duration_s, config=TrackingConfig(loop_hz=20.0, error_log_hz=20.0, mount_lag_s=lag_s),
                on_tick=ticks.append,
            )
        finally:
            mount.stop()
            safety.shutdown()
            mount.close()
        return ticks

    ticks_unshifted = run(0.0)
    ticks_shifted = run(mount_lag_s)

    rate_deg_s = rate_x * SIDEREAL_DEG_PER_S
    expected_shift_arcsec = rate_deg_s * mount_lag_s * 3600
    actual_shift_arcsec = (ticks_shifted[-1]["target_ra_deg"] - ticks_unshifted[-1]["target_ra_deg"]) * 3600
    assert actual_shift_arcsec == pytest.approx(expected_shift_arcsec, abs=20.0)


def test_tracking_loop_feedback_reduces_along_track_error_vs_feedforward_only():
    # Closed-loop regression test locking in the empirically-verified
    # behavior: with enable_feedback=True, the PI trim should measurably
    # shrink the steady-state along-track lag over the run, compared to
    # feedforward-only. Seeded MockMount for determinism (its simulated
    # serial latency is Gaussian-jittered otherwise).
    start_ra, start_dec = 10.0, 45.0
    rate_x = 50.0
    duration_s = 4.0

    def run(enable_feedback):
        trajectory = _make_constant_rate_trajectory(rate_x, duration_s + 1.0, start_ra, start_dec)
        mock = MockMount(
            MockConfig(rv_mode="per_axis", tracking_adds=True, start_ra_deg=start_ra, start_dec_deg=start_dec),
            seed=1,
        )
        mount = Mount(mock)
        safety = SafetyGuard(mount, watchdog_timeout=5.0)
        fh = io.StringIO()
        writer = csv.DictWriter(fh, fieldnames=TRACKING_CSV_FIELDS)
        writer.writeheader()
        ticks = []
        try:
            run_tracking_loop(
                mount, safety, trajectory, AxisSigns(ra=1.0, dec=1.0), LiveOffsets(), writer,
                duration_s=duration_s, config=TrackingConfig(loop_hz=20.0, enable_feedback=enable_feedback),
                on_tick=ticks.append,
            )
        finally:
            mount.stop()
            safety.shutdown()
            mount.close()
        return ticks

    ticks_off = run(False)
    ticks_on = run(True)

    final_error_off = abs(ticks_off[-1]["along_track_arcsec"])
    final_error_on = abs(ticks_on[-1]["along_track_arcsec"])
    # Observed gap is consistently ~10-14" across runs; require at least
    # half that margin so ordinary timing jitter doesn't flake the test.
    assert final_error_on < final_error_off - 5.0




def _ra_deg_at_ha(ha_deg: float, longitude_deg: float) -> float:
    """The RA (degrees) that puts the mount's hour angle at ha_deg RIGHT
    NOW, for the given site longitude -- see test_mount.py's own copy of
    this helper for the full reasoning (lets a test start already however
    far past/before the meridian it needs, instead of waiting real minutes
    for sidereal time to get there on its own)."""
    lst_deg = (gmst_deg(datetime.now(timezone.utc)) + longitude_deg) % 360.0
    return (lst_deg - ha_deg) % 360.0


def test_meridian_track_limit_deg_defaults_to_none():
    # Deliberately conservative default -- see TrackingConfig.meridian_
    # track_limit_deg's own docstring: Mount.set_meridian_behavior's reply
    # format is unconfirmed against real hardware, and a wrong assumption
    # there desyncs the ENTIRE session's protocol stream, not just this
    # one feature -- so it must not run unattended before being verified
    # once by hand.
    assert TrackingConfig().meridian_track_limit_deg is None


def test_run_tracking_loop_configures_generous_meridian_tracking_before_the_pass_when_enabled():
    # The actual fix for a real "tracking diverges badly right after the
    # meridian" incident (see TrackingConfig.meridian_track_limit_deg's
    # docstring): confirm the loop itself issues :ST# with a generous,
    # non-stopping configuration before committing to a pass, instead of
    # relying on whatever the mount's own factory/current default is --
    # only once explicitly enabled (not the default, see the test above).
    mock = MockMount(MockConfig(
        rv_mode="per_axis", meridian_track_past=False, meridian_limit_deg=0.0,
        start_ra_deg=10.0, start_dec_deg=45.0,
    ))
    mount = Mount(mock)
    safety = SafetyGuard(mount, watchdog_timeout=5.0)
    trajectory = _make_constant_rate_trajectory(0.0, 1.0, 10.0, 45.0)
    fh = io.StringIO()
    writer = csv.DictWriter(fh, fieldnames=TRACKING_CSV_FIELDS)
    writer.writeheader()
    try:
        run_tracking_loop(
            mount, safety, trajectory, AxisSigns(ra=1.0, dec=1.0), LiveOffsets(), writer,
            duration_s=0.1, config=TrackingConfig(loop_hz=20.0, meridian_track_limit_deg=15.0),
        )
        assert mount.get_meridian_behavior() == (False, True, 15.0)
    finally:
        mount.stop()
        safety.shutdown()
        mount.close()


def test_run_tracking_loop_skips_meridian_configuration_when_set_to_none():
    mock = MockMount(MockConfig(
        rv_mode="per_axis", meridian_track_past=False, meridian_limit_deg=3.0,
        start_ra_deg=10.0, start_dec_deg=45.0,
    ))
    mount = Mount(mock)
    safety = SafetyGuard(mount, watchdog_timeout=5.0)
    trajectory = _make_constant_rate_trajectory(0.0, 1.0, 10.0, 45.0)
    fh = io.StringIO()
    writer = csv.DictWriter(fh, fieldnames=TRACKING_CSV_FIELDS)
    writer.writeheader()
    try:
        run_tracking_loop(
            mount, safety, trajectory, AxisSigns(ra=1.0, dec=1.0), LiveOffsets(), writer,
            duration_s=0.1, config=TrackingConfig(loop_hz=20.0, meridian_track_limit_deg=None),
        )
        assert mount.get_meridian_behavior() == (False, False, 3.0)
    finally:
        mount.stop()
        safety.shutdown()
        mount.close()


def test_run_tracking_loop_calls_on_limit_warning_when_mount_reports_a_limit_code():
    # Regression: :GAT# limit codes (5/6/8) used to only ever reach a
    # stderr print() -- invisible to the GUI unless launched from a terminal
    # with stderr visible. on_limit_warning is what MountWorker wires to an
    # actual "log" event the GUI's log panel shows -- see
    # am5/gui/worker.py's _handle_start_tracking.
    longitude_deg = 6.14
    start_ra_deg = _ra_deg_at_ha(5.0, longitude_deg)  # already 5 deg past the meridian
    cfg = MockConfig(
        rv_mode="per_axis", start_ra_deg=start_ra_deg, start_dec_deg=45.0,
        longitude_deg=longitude_deg, meridian_limit_enabled=True, meridian_track_past=False,
    )
    mock = MockMount(cfg)
    mount = Mount(mock)
    safety = SafetyGuard(mount, watchdog_timeout=5.0)
    trajectory = _make_constant_rate_trajectory(0.0, 2.0, start_ra_deg, 45.0)
    fh = io.StringIO()
    writer = csv.DictWriter(fh, fieldnames=TRACKING_CSV_FIELDS)
    writer.writeheader()
    warnings = []
    try:
        # meridian_track_limit_deg=None -- don't let the loop's own fix
        # reconfigure the mount away from the stuck state this test needs.
        run_tracking_loop(
            mount, safety, trajectory, AxisSigns(ra=1.0, dec=1.0), LiveOffsets(), writer,
            duration_s=1.2,
            config=TrackingConfig(loop_hz=20.0, status_check_hz=5.0, meridian_track_limit_deg=None),
            on_limit_warning=warnings.append,
        )
    finally:
        mount.stop()
        safety.shutdown()
        mount.close()
    assert warnings == [8]
