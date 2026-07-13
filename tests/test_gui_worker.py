import queue
import time
from unittest.mock import patch

import pytest

from am5.clock_sync import ClockSyncStatus
from am5.gui.worker import MountWorker, WorkerEvent
from am5.mock_mount import MockConfig, MockMount
from am5.mount import Mount
from am5.safety import SafetyGuard
from am5.tracker import AxisSigns


def _wait_for(worker: MountWorker, kind: str, timeout: float = 5.0) -> WorkerEvent:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = worker.events.get(timeout=0.1)
        except queue.Empty:
            continue
        if event.kind == kind:
            return event
    raise AssertionError(f"never saw a {kind!r} event within {timeout}s")


def _collect_until(worker: MountWorker, kind: str, timeout: float = 5.0) -> list[WorkerEvent]:
    """Like _wait_for, but returns every event seen along the way (in
    order) instead of discarding non-matches -- for asserting about events
    that must arrive *before* a given one."""
    seen: list[WorkerEvent] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = worker.events.get(timeout=0.1)
        except queue.Empty:
            continue
        seen.append(event)
        if event.kind == kind:
            return seen
    raise AssertionError(f"never saw a {kind!r} event within {timeout}s (saw: {[e.kind for e in seen]})")


@pytest.fixture
def worker():
    w = MountWorker()
    yield w
    w.shutdown()


def test_connect_emits_connected_with_firmware(worker):
    worker.connect("mock", mock_seed=1)
    event = _wait_for(worker, "connected")
    assert "firmware" in event.payload


def test_connect_warns_in_log_when_system_clock_not_synchronized(worker):
    # sync_site_and_time (called during connect) pushes THIS machine's
    # clock to the mount -- if it's wrong, the mount's time is wrong too,
    # with no way to check that afterward (see _handle_connect's comment).
    # Connecting must still succeed (soft warning only).
    status = ClockSyncStatus(synchronized=False, offset_s=2.5, source="chronyc tracking", detail="...")
    with patch("am5.gui.worker.check_clock_sync", return_value=status):
        worker.connect("mock", mock_seed=1)
        seen = _collect_until(worker, "connected", timeout=3.0)
    log_events = [e for e in seen if e.kind == "log"]
    assert any("NOT synchronized" in e.payload["message"] for e in log_events)
    assert any("2.50s" in e.payload["message"] for e in log_events)


def test_connect_warns_in_log_when_clock_sync_status_is_unknown(worker):
    status = ClockSyncStatus(synchronized=None, offset_s=None, source="none", detail="no clock-sync tool available")
    with patch("am5.gui.worker.check_clock_sync", return_value=status):
        worker.connect("mock", mock_seed=1)
        seen = _collect_until(worker, "connected", timeout=3.0)
    log_events = [e for e in seen if e.kind == "log"]
    assert any("could not determine" in e.payload["message"] for e in log_events)


def test_connect_does_not_warn_when_system_clock_is_synchronized(worker):
    status = ClockSyncStatus(synchronized=True, offset_s=0.001, source="timedatectl timesync-status", detail="...")
    with patch("am5.gui.worker.check_clock_sync", return_value=status):
        worker.connect("mock", mock_seed=1)
        seen = _collect_until(worker, "connected", timeout=3.0)
    log_events = [e for e in seen if e.kind == "log"]
    assert not any("synchroni" in e.payload["message"].lower() for e in log_events)


def test_idle_poll_emits_position_after_connect(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    event = _wait_for(worker, "position", timeout=3.0)
    assert "ra_hours" in event.payload
    assert "dec_deg" in event.payload


def test_jog_moves_the_mount(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    first = _wait_for(worker, "position", timeout=3.0)

    worker.jog_start("e", rate_x=100.0)
    time.sleep(0.5)
    worker.jog_stop("e")

    moved = False
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            event = worker.events.get(timeout=0.2)
        except queue.Empty:
            continue
        if event.kind == "position" and abs(event.payload["ra_hours"] - first.payload["ra_hours"]) > 1e-4:
            moved = True
            break
    assert moved


def test_goto_reports_result_and_arrival(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    _wait_for(worker, "position", timeout=3.0)

    # circumpolar from the mock's default +46.18 latitude -- deterministically above horizon
    worker.goto(ra_hours=12.0, dec_deg=80.0)
    result = _wait_for(worker, "goto_result", timeout=3.0)
    assert result.payload["code"] == 0
    # the requested target must be echoed back verbatim -- this is what lets
    # an operator confirm the same target was actually sent on a repeat click
    assert result.payload["target_ra_hours"] == pytest.approx(12.0)
    assert result.payload["target_dec_deg"] == pytest.approx(80.0)
    arrived = _wait_for(worker, "goto_arrived", timeout=5.0)
    assert arrived.payload["ra_hours"] == pytest.approx(12.0, abs=1e-2)
    assert arrived.payload["dec_deg"] == pytest.approx(80.0, abs=1e-2)


def test_sync_reports_result_and_updates_position_without_moving(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    before = _wait_for(worker, "position", timeout=3.0)
    assert before.payload["ra_hours"] == pytest.approx(3.0)  # mock's default start

    worker.sync(ra_hours=5.5, dec_deg=20.0)
    result = _wait_for(worker, "sync_result")
    assert result.payload["ok"] is True
    assert result.payload["ra_hours"] == pytest.approx(5.5)
    assert result.payload["dec_deg"] == pytest.approx(20.0)

    # position jumps to the synced target instantly -- no slew, no
    # intermediate positions, unlike a goto
    after = _wait_for(worker, "position", timeout=3.0)
    assert after.payload["ra_hours"] == pytest.approx(5.5)
    assert after.payload["dec_deg"] == pytest.approx(20.0)


def test_sync_before_connect_is_a_silent_no_op(worker):
    worker.sync(ra_hours=5.5, dec_deg=20.0)
    time.sleep(0.3)
    assert worker.events.empty()


def test_goto_below_horizon_is_rejected_without_moving(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    _wait_for(worker, "position", timeout=3.0)

    worker.goto(ra_hours=12.0, dec_deg=-89.0)
    result = _wait_for(worker, "goto_result", timeout=3.0)
    assert result.payload["code"] == 1
    assert "below horizon" in result.payload["meaning"]


def test_jog_goto_aborts_on_divergence_from_wrong_axis_sign(worker):
    # A wrong DEC sign makes the controller drive the DEC axis away from
    # the target -- must abort quickly, not jog the wrong way for 180s.
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    _wait_for(worker, "position", timeout=3.0)

    # mock 'n' increases DEC; telling the controller dec sign is -1 inverts it
    worker.jog_goto(ra_hours=3.0, dec_deg=60.0, axis_signs=AxisSigns(ra=1.0, dec=-1.0))
    result = _wait_for(worker, "jog_goto_result", timeout=20.0)
    assert result.payload["arrived"] is False


def test_jog_goto_converges_over_a_large_initial_separation(worker):
    # Real incident this session: a GOTO to Deneb (~57 deg away, from
    # near the pole) tripped the divergence guard despite correct
    # calibration and no pier flip. Root cause was two compounding bugs,
    # both fixed together (see angular_separation_deg's docstring and the
    # rate-synchronization comment in _handle_jog_goto): the guard's
    # error metric was a small-angle approximation invalid at this
    # distance, and independently rate-capping each axis let DEC (the
    # smaller raw error, near the pole) race ahead of RA, visiting a
    # temporarily-worse great-circle path.
    mock = MockMount(MockConfig(rv_mode="per_axis", tracking_adds=True, start_ra_deg=196.5, start_dec_deg=70.59))
    mount = Mount(mock)
    safety = SafetyGuard(mount, watchdog_timeout=5.0, install_signal_handlers=False)
    worker._mount = mount
    worker._safety = safety
    axis_signs = AxisSigns(ra=1.0, dec=1.0, calibrated_pier_side=mount.get_pier_side())

    try:
        worker._handle_jog_goto({"ra_hours": 310.357973 / 15.0, "dec_deg": 45.280334, "axis_signs": axis_signs})
    finally:
        mount.stop()
        safety.shutdown()

    result = _wait_for(worker, "jog_goto_result", timeout=1.0)
    assert result.payload["arrived"] is True


def test_jog_goto_converges_without_using_ms(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    _wait_for(worker, "position", timeout=3.0)

    # mock's default start is RA=3h DEC=45deg -- pick a nearby target so the
    # proportional jog controller converges quickly in the test.
    worker.jog_goto(ra_hours=3.2, dec_deg=46.0, axis_signs=AxisSigns(ra=1.0, dec=1.0))
    result = _wait_for(worker, "jog_goto_result", timeout=15.0)
    assert result.payload["arrived"] is True

    radec = None
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            event = worker.events.get(timeout=0.2)
        except queue.Empty:
            continue
        if event.kind == "position":
            radec = event.payload
    assert radec is not None
    assert radec["ra_hours"] == pytest.approx(3.2, abs=0.01)
    assert radec["dec_deg"] == pytest.approx(46.0, abs=0.01)


def test_emergency_stop_aborts_a_running_jog_goto(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    _wait_for(worker, "position", timeout=3.0)

    # a far target so the jog controller is still running (not yet arrived)
    # when we hit emergency stop -- must abort promptly, not run to the
    # 180s timeout or ignore the stop.
    worker.jog_goto(ra_hours=15.0, dec_deg=-80.0, axis_signs=AxisSigns(ra=1.0, dec=1.0))
    time.sleep(0.5)  # let the loop get going
    worker.emergency_stop()
    result = _wait_for(worker, "jog_goto_result", timeout=3.0)
    assert result.payload["arrived"] is False


def test_emergency_stop_ends_a_tracking_pass(worker, tmp_path):
    import numpy as np

    from am5.ephemeris import Trajectory
    from am5.tracker import AxisSigns as _AxisSigns
    from am5.tracker import LiveOffsets

    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")

    n = 6000
    t_unix = time.time() + np.linspace(0, 120, n)
    traj = Trajectory(
        t_unix=t_unix, ra_deg=np.full(n, 45.0), dec_deg=np.full(n, 45.0),
        dra_dt_deg_s=np.full(n, 0.01), ddec_dt_deg_s=np.full(n, 0.01),
        alt_deg=np.full(n, 45.0), az_deg=np.full(n, 180.0), ha_hours=np.zeros(n),
        distance_km=np.full(n, 500.0),
    )
    worker.start_tracking(traj, _AxisSigns(ra=1.0, dec=1.0), LiveOffsets(), tmp_path / "t.csv", duration_s=60.0)
    _wait_for(worker, "tracking_started", timeout=3.0)
    worker.emergency_stop()
    _wait_for(worker, "tracking_stopped", timeout=3.0)  # emergency stop must end the pass, not just briefly halt it


def test_goto_mismatch_check_warns_when_landed_far_from_target(worker):
    # Reproduces the real-hardware incident: :MS# replies success and the
    # position genuinely stops changing, but degrees away from what was
    # requested -- must not be silently accepted as "arrived".
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker._check_goto_landed_on_target(12.0, 80.0, (9.7228, 45.5775))
    log_event = _wait_for(worker, "log", timeout=2.0)
    assert "landed" in log_event.payload["message"]


def test_goto_mismatch_check_silent_when_close_to_target(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker._check_goto_landed_on_target(12.0, 80.0, (12.0003, 80.001))

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            event = worker.events.get(timeout=0.1)
        except queue.Empty:
            continue
        if event.kind == "log":
            assert "landed" not in event.payload.get("message", "")


def test_calibrate_emits_axis_signs(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    _wait_for(worker, "position", timeout=3.0)

    worker.calibrate()
    event = _wait_for(worker, "calibration_done", timeout=5.0)
    assert event.payload["ra_sign"] in (1.0, -1.0)
    assert event.payload["dec_sign"] in (1.0, -1.0)
    # dec_sign is only valid for this pier side -- see AxisSigns'
    # docstring -- so the event has to carry it for App to track flips.
    assert event.payload["pier_side"] in ("E", "W", "N", None)


def test_measure_mount_lag_emits_result(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    _wait_for(worker, "position", timeout=3.0)

    worker.measure_mount_lag(rate_x=100.0, duration_s=0.5)
    event = _wait_for(worker, "mount_lag_result", timeout=5.0)
    assert event.payload["lag_s"] >= 0.0
    assert event.payload["samples"] > 0


def test_disconnect_emits_disconnected(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.disconnect()
    _wait_for(worker, "disconnected", timeout=3.0)


def test_park_sends_mount_home_and_emits_parked(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.goto(ra_hours=12.0, dec_deg=80.0)
    _wait_for(worker, "goto_arrived", timeout=5.0)

    worker.park()
    _wait_for(worker, "parked", timeout=3.0)

    position = _wait_for(worker, "position", timeout=3.0)
    assert position.payload["ra_hours"] == pytest.approx(3.0, abs=1e-2)  # mock's default start_ra_deg=45 -> 3h
    assert position.payload["dec_deg"] == pytest.approx(45.0, abs=1e-2)


def test_park_native_emits_parked_with_method_and_reply(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")

    worker.park_native()
    event = _wait_for(worker, "parked", timeout=3.0)
    assert event.payload["method"] == "native"
    assert event.payload["reply"] == "1#"


def test_jog_is_blocked_while_parked(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.park()
    _wait_for(worker, "parked", timeout=3.0)

    baseline = _wait_for(worker, "position", timeout=3.0)
    worker.jog_start("e", rate_x=200.0)

    # the blocked-command warning must show up before any further draining,
    # since _wait_for discards non-matching events as it scans
    log_event = _wait_for(worker, "log", timeout=3.0)
    assert "parked" in log_event.payload["message"]

    time.sleep(0.5)
    worker.jog_stop("e")
    pos = _wait_for(worker, "position", timeout=3.0)
    assert pos.payload["ra_hours"] == pytest.approx(baseline.payload["ra_hours"], abs=1e-4)


def test_unpark_restores_jog(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.park()
    _wait_for(worker, "parked", timeout=3.0)

    worker.unpark()
    _wait_for(worker, "unparked", timeout=3.0)

    baseline = _wait_for(worker, "position", timeout=3.0)
    worker.jog_start("e", rate_x=200.0)
    time.sleep(0.5)
    worker.jog_stop("e")

    moved = False
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            event = worker.events.get(timeout=0.2)
        except queue.Empty:
            continue
        if event.kind == "position" and abs(event.payload["ra_hours"] - baseline.payload["ra_hours"]) > 1e-4:
            moved = True
            break
    assert moved


def test_park_native_refused_in_altaz_mode(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    assert worker._mount is not None
    worker._mount._send(b":AA#", expect_response=False)  # switch to Alt-Az mode

    worker.park_native()
    log_event = _wait_for(worker, "log", timeout=3.0)
    assert "Alt-Az" in log_event.payload["message"]


def test_unpark_warns_if_p_flag_still_set(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.park_native()
    _wait_for(worker, "parked", timeout=3.0)

    # the mock has no real unpark side effect, so :GU# still reports P --
    # exactly the scenario the warning exists to catch on real hardware.
    worker.unpark()
    log_event = _wait_for(worker, "log", timeout=3.0)
    assert "P" in log_event.payload["message"]
    _wait_for(worker, "unparked", timeout=3.0)


def test_goto_is_blocked_while_parked(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.park()
    _wait_for(worker, "parked", timeout=3.0)

    worker.goto(ra_hours=12.0, dec_deg=80.0)
    log_event = _wait_for(worker, "log", timeout=3.0)
    assert "parked" in log_event.payload["message"]
