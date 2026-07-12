import pytest

from am5.mock_mount import MockConfig, MockMount
from am5.mount import Mount


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
