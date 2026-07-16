import time
from datetime import datetime, timezone

import pytest

from am5.angles import gmst_deg
from am5.constants import SIDEREAL_DEG_PER_S
from am5.mock_mount import MockConfig, MockMount
from am5.mount import Mount
from am5.protocol import parse_error


@pytest.fixture
def mount() -> Mount:
    m = Mount(MockMount(MockConfig(start_ra_deg=45.0, start_dec_deg=45.0)))
    m.sync_site_and_time(46.18, 6.14)
    yield m
    m.close()


@pytest.fixture
def unsynced_mount() -> Mount:
    m = Mount(MockMount(MockConfig(start_ra_deg=45.0, start_dec_deg=45.0)))
    yield m
    m.close()


def test_set_tracking_reads_a_bare_single_char_reply_not_hash_terminated(mount):
    # Regression: :Te#/:Td# reply "1"/"0" with NO '#' terminator (confirmed
    # against the protocol doc and real hardware) -- set_tracking() used to
    # go through _send()'s read_until_hash, which blocked for the full
    # response_timeout (2s on real hardware) on every single call since
    # that '#' never arrives. Both the mock (which used to queue "1#"/"0#",
    # masking this) and Mount.set_tracking() itself were fixed together.
    assert mount.set_tracking(True) == "1"
    assert mount.set_tracking(False) == "1"


def test_set_altitude_limits_enabled_reads_a_bare_single_char_reply(mount):
    # Same bug class as set_tracking() -- :SLE#/:SLD# also reply "1"/"0"
    # with no '#' terminator (confirmed on real hardware: 2s stall per
    # call before this fix).
    assert mount.set_altitude_limits_enabled(True) == "1"
    assert mount.set_altitude_limits_enabled(False) == "1"


def test_sync_site_and_time_reads_bare_single_char_replies(mount):
    # Same bug class again -- :SMTI#/:St#/:Sg# all reply "1"/"0" with no
    # '#' terminator. Confirmed on real hardware: this was costing 6
    # *full seconds* (2s x 3 commands) on every single connect before this
    # fix, since read_until_hash blocked for the whole response_timeout on
    # each of the three sub-commands.
    mount.sync_site_and_time(46.18, 6.14)  # must not raise ProtocolError


def test_goto_moves_to_target(mount):
    mount.set_tracking(False)
    # DEC=+80 is circumpolar (always above the horizon) from the mock's
    # default +46.18 latitude, regardless of sidereal time at test-run
    # time — DEC=-10 would flakily pass or fail depending on when the test
    # happens to run relative to the mock's internal clock.
    result = mount.goto(ra_hours=12.0, dec_deg=80.0)
    assert result == 0
    radec = mount.get_radec()
    assert radec.ra_hours == pytest.approx(12.0, abs=1e-3)
    assert radec.dec_deg == pytest.approx(80.0, abs=1e-3)


def test_goto_rejects_target_below_horizon(mount):
    # DEC=-89 never rises above the horizon from the mock's default +46.18
    # latitude, regardless of sidereal time -- a deterministically
    # below-horizon target.
    baseline = mount.get_radec()
    result = mount.goto(ra_hours=12.0, dec_deg=-89.0)
    assert result == 1
    radec = mount.get_radec()
    assert radec.ra_hours == pytest.approx(baseline.ra_hours, abs=1e-3)
    assert radec.dec_deg == pytest.approx(baseline.dec_deg, abs=1e-3)


def test_set_target_ra_dec_individually(mount):
    assert mount.set_target_ra(6.0) is True
    assert mount.set_target_dec(20.0) is True


def test_sync_sets_position_without_reply_ambiguity(mount):
    reply = mount.sync(ra_hours=6.0, dec_deg=30.0)
    assert reply.strip() == "N/A#"
    radec = mount.get_radec()
    assert radec.ra_hours == pytest.approx(6.0, abs=1e-3)
    assert radec.dec_deg == pytest.approx(30.0, abs=1e-3)


def test_park_returns_to_home_position(mount):
    mount.set_tracking(False)
    mount.goto(ra_hours=6.0, dec_deg=80.0)
    mount.park()
    radec = mount.get_radec()
    assert radec.ra_hours == pytest.approx(45.0 / 15.0, abs=1e-3)
    assert radec.dec_deg == pytest.approx(45.0, abs=1e-3)


def test_park_native_replies_and_moves_to_a_different_position_than_park(mount):
    mount.set_tracking(False)
    reply = mount.park_native()
    assert reply.strip() == "1#"
    radec = mount.get_radec()
    # mock's placeholder :hP# position -- not a claim about real hardware,
    # just needs to be distinct from park()'s :hC# home so the two are
    # visually comparable in --mock mode.
    assert radec.ra_hours == pytest.approx(6.0, abs=1e-3)
    assert radec.dec_deg == pytest.approx(0.0, abs=1e-3)


def test_get_pier_side_returns_bare_letter():
    m = Mount(MockMount(MockConfig(start_ra_deg=45.0, start_dec_deg=45.0)))
    try:
        m.park()  # :hC# -> at_home -> mock reports 'N'
        assert m.get_pier_side() == "N"
    finally:
        m.close()


def test_park_native_refused_outside_equatorial_mode(mount):
    mount._send(b":AA#", expect_response=False)  # switch to Alt-Az mode
    reply = mount.park_native()
    assert reply.strip() == "0#"
    status = mount.get_status()
    assert status.is_parked is False


def test_goto_without_sync_is_rejected_with_e7(unsynced_mount):
    result = unsynced_mount.goto(ra_hours=12.0, dec_deg=80.0)
    assert result == -7


def test_sync_site_and_time_then_status_reports_equatorial_and_not_parked(mount):
    status = mount.get_status()
    assert status.is_equatorial is True
    assert status.is_parked is False


def test_get_status_reflects_park_and_home():
    m = Mount(MockMount(MockConfig(start_ra_deg=45.0, start_dec_deg=45.0)))
    try:
        m.park()
        assert m.get_status().is_at_home is True
        m.park_native()
        status = m.get_status()
        assert status.is_parked is True
        assert status.is_at_home is False
    finally:
        m.close()


def _ra_deg_at_ha(ha_deg: float, longitude_deg: float) -> float:
    """The RA (degrees) that puts the mount's hour angle at ha_deg RIGHT
    NOW, for the given site longitude -- lets a test start already
    however far past (or before) the meridian it needs, instead of
    waiting real minutes for sidereal time to get there on its own."""
    lst_deg = (gmst_deg(datetime.now(timezone.utc)) + longitude_deg) % 360.0
    return (lst_deg - ha_deg) % 360.0


def test_meridian_behavior_roundtrip_through_the_mock():
    m = Mount(MockMount(MockConfig(start_ra_deg=45.0, start_dec_deg=45.0)))
    try:
        m.set_meridian_behavior(track_past_meridian=True, limit_deg=12.0, flip=False)
        assert m.get_meridian_behavior() == (False, True, 12.0)
    finally:
        m.close()


def test_inject_pointing_error_offsets_ra_and_dec():
    # MockMount.inject_pointing_error has no wire-protocol equivalent --
    # it's the training-scenario aid TransitPanel's "Simulate a random
    # pointing error" checkbox uses (see am5/gui/panels.py's
    # _inject_training_pointing_error) to make the mock ISS start off-
    # centre, forcing the operator to actually use the finder to re-
    # acquire it, same as a real residual GOTO/polar-alignment error would.
    mock = MockMount(MockConfig(start_ra_deg=45.0, start_dec_deg=45.0))
    m = Mount(mock)
    try:
        m.sync_site_and_time(46.18, 6.14)
        before = m.get_radec()
        mock.inject_pointing_error(ra_bias_deg=2.0, dec_bias_deg=-1.0)
        after = m.get_radec()
        assert after.ra_hours == pytest.approx(before.ra_hours + 2.0 / 15.0, abs=1e-6)
        assert after.dec_deg == pytest.approx(before.dec_deg - 1.0, abs=1e-6)
    finally:
        m.close()


def test_inject_pointing_error_wraps_ra_and_clamps_dec_at_the_poles():
    mock = MockMount(MockConfig(start_ra_deg=359.0, start_dec_deg=89.0))
    m = Mount(mock)
    try:
        m.sync_site_and_time(46.18, 6.14)
        mock.inject_pointing_error(ra_bias_deg=5.0, dec_bias_deg=5.0)
        after = m.get_radec()
        assert after.ra_hours == pytest.approx((364.0 % 360.0) / 15.0, abs=1e-6)
        assert after.dec_deg == pytest.approx(90.0, abs=1e-6)
    finally:
        m.close()


def test_mock_mount_configured_to_stop_freezes_ra_shortly_past_the_meridian():
    # Regression: this project never sent :ST# before, so a real mount ran
    # on whatever its own factory/current default meridian behavior is --
    # explicitly configuring the mock into the documented worst case (stop
    # 1 degree past the meridian) reproduces the real "tracking diverges
    # right after the meridian" incident against the mock, without needing
    # hardware to see the symptom happen at all. NOT MockConfig's own
    # default (that defaults permissive -- see its docstring for why a
    # wall-clock-dependent default broke an unrelated test).
    longitude_deg = 6.14
    start_ra_deg = _ra_deg_at_ha(5.0, longitude_deg)  # already 5 deg past the meridian
    cfg = MockConfig(
        start_ra_deg=start_ra_deg, start_dec_deg=45.0, longitude_deg=longitude_deg,
        meridian_limit_enabled=True, meridian_track_past=False,
    )
    m = Mount(MockMount(cfg))
    try:
        m.sync_site_and_time(46.18, longitude_deg)
        m.set_tracking(True)
        time.sleep(0.1)  # a few 200Hz sim-loop ticks
        assert parse_error(m.get_tracking_status()) == 8
        ra_before = m.get_radec().ra_hours
        m.set_rate(300.0)  # a large rate -- if this moved anything, it'd be unmissable
        m.move("e")
        time.sleep(0.2)
        # The mount silently ignores further rate commands once stopped --
        # this is the exact mechanism that made run_tracking_loop's own
        # error-tracking diverge in the real incident.
        assert m.get_radec().ra_hours == pytest.approx(ra_before, abs=1e-4)
    finally:
        m.close()


def test_set_meridian_behavior_to_continue_tracking_prevents_the_stop():
    # The fix: explicitly configuring the mount to keep tracking past the
    # meridian (up to the protocol's own max, 15 deg) before a pass means
    # a normal few-degree crossing never trips the silent stop.
    longitude_deg = 6.14
    start_ra_deg = _ra_deg_at_ha(5.0, longitude_deg)  # 5 deg past -- within a 15 deg allowance
    cfg = MockConfig(start_ra_deg=start_ra_deg, start_dec_deg=45.0, longitude_deg=longitude_deg, meridian_limit_enabled=True)
    m = Mount(MockMount(cfg))
    try:
        m.sync_site_and_time(46.18, longitude_deg)
        m.set_meridian_behavior(track_past_meridian=True, limit_deg=15.0)
        m.set_tracking(True)
        time.sleep(0.1)
        assert parse_error(m.get_tracking_status()) is None
        ra_before = m.get_radec().ra_hours
        m.set_rate(300.0)
        m.move("e")
        time.sleep(0.2)
        # Rate commands still take effect -- RA actually moved, unlike the
        # stopped case above.
        assert m.get_radec().ra_hours != pytest.approx(ra_before, abs=1e-4)
        assert parse_error(m.get_tracking_status()) is None
    finally:
        m.close()


def test_meridian_stop_only_freezes_ra_not_dec():
    # Per the protocol doc's own note: only "the RA axis... continues to
    # track the angle of rotation" up to the limit -- DEC is unaffected.
    longitude_deg = 6.14
    start_ra_deg = _ra_deg_at_ha(5.0, longitude_deg)
    cfg = MockConfig(
        start_ra_deg=start_ra_deg, start_dec_deg=0.0, longitude_deg=longitude_deg,
        meridian_limit_enabled=True, meridian_track_past=False,
    )
    m = Mount(MockMount(cfg))
    try:
        m.sync_site_and_time(46.18, longitude_deg)
        m.set_tracking(True)
        time.sleep(0.1)
        assert parse_error(m.get_tracking_status()) == 8
        m.set_rate(300.0)
        m.move("n")
        time.sleep(0.2)
        assert m.get_radec().dec_deg > 0.05
    finally:
        m.close()


def test_set_rate_ra_and_dec_are_independent(mount):
    # Regression: confirmed live on real AM5 hardware (docs/AM5_UART_
    # protocol_1.8.8.md, not in the official v1.7 PDF) that :Rvr#/:Rvd#
    # store RA/DEC manual speed independently -- changing one must never
    # affect the other's already-latched motion, unlike plain :Rv# (which
    # writes both axes' stored speed at once).
    mount.set_rate_ra(150.0)
    mount.set_rate_dec(60.0)
    mount.move("e")
    mount.move("n")
    time.sleep(0.2)
    pos0 = mount.get_radec()

    mount.set_rate_ra(250.0)  # must not touch DEC's stored 60.0
    time.sleep(0.2)
    pos1 = mount.get_radec()
    ra_speed_arcsec_s = abs(pos1.ra_hours - pos0.ra_hours) * 15 * 3600 / 0.2
    dec_speed_arcsec_s = abs(pos1.dec_deg - pos0.dec_deg) * 3600 / 0.2
    sidereal_arcsec_s = SIDEREAL_DEG_PER_S * 3600.0
    # DEC should still be moving at ~60x, RA not yet at 250x (needs a
    # fresh :Me# to re-latch, same per-axis-latch model as plain :Rv#).
    assert dec_speed_arcsec_s == pytest.approx(60.0 * sidereal_arcsec_s, rel=0.3)
    assert ra_speed_arcsec_s < 200.0 * sidereal_arcsec_s

    mount.move("e")  # re-latch RA onto the new 250x
    time.sleep(0.2)
    pos2 = mount.get_radec()
    ra_speed_arcsec_s = abs(pos2.ra_hours - pos1.ra_hours) * 15 * 3600 / 0.2
    assert ra_speed_arcsec_s == pytest.approx(250.0 * sidereal_arcsec_s, rel=0.3)


def test_health_diagnostic_reads(mount):
    assert mount.get_axis_stall_load("ra") == 0
    assert mount.get_axis_stall_load("dec") == 0
    assert mount.get_temperature_c() == pytest.approx(25.0)
    assert mount.get_max_rate_x() == pytest.approx(1440.0)

    # Motor current rises while an axis is actively driven vs. resting --
    # confirmed live on real AM5 hardware (15 -> 28 on RA while jogging).
    assert mount.get_axis_current("ra") == 15
    mount.set_rate_ra(100.0)
    mount.move("e")
    assert mount.get_axis_current("ra") == 28
    mount.stop("e")


def test_alignment_mode_accumulates_sync_points(mount):
    assert mount.get_alignment_mode() is False
    assert mount.get_alignment_point_count() == 0

    reply = mount.set_alignment_mode(True)
    assert reply.strip().rstrip("#") == "1"
    assert mount.get_alignment_mode() is True

    mount.set_target_ra(1.0)
    mount.set_target_dec(45.0)
    mount.sync_to_target()
    mount.set_target_ra(2.0)
    mount.set_target_dec(50.0)
    mount.sync_to_target()
    assert mount.get_alignment_point_count() == 2


def test_alignment_mode_off_clears_the_point_count(mount):
    mount.set_alignment_mode(True)
    mount.set_target_ra(1.0)
    mount.set_target_dec(45.0)
    mount.sync_to_target()
    assert mount.get_alignment_point_count() == 1

    mount.set_alignment_mode(False)
    assert mount.get_alignment_point_count() == 0
    # Re-enabling starts a fresh table, not a resumed one.
    mount.set_alignment_mode(True)
    assert mount.get_alignment_point_count() == 0


def test_alignment_mode_sync_does_not_move_reported_position(mount):
    # Documented as an unconfirmed guess (see _MountState.alignment_mode's
    # own comment) -- locking in the mock's own conservative choice so a
    # future change to it is a deliberate decision, not silent drift.
    before = mount.get_radec()
    mount.set_alignment_mode(True)
    mount.set_target_ra(before.ra_hours + 5.0)
    mount.set_target_dec(before.dec_deg + 10.0)
    mount.sync_to_target()
    after = mount.get_radec()
    assert after.ra_hours == pytest.approx(before.ra_hours, abs=1e-6)
    assert after.dec_deg == pytest.approx(before.dec_deg, abs=1e-6)
