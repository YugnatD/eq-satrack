import csv
import io

from am5.mock_mount import MockConfig, MockMount
from am5.mount import Mount
from characterize import poll_radec


class _RecordingSafety:
    def __init__(self):
        self.calls: list[bool] = []

    def notify_command(self, movement_active: bool):
        self.calls.append(movement_active)


def _csv_writer():
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["t_mono", "t_utc", "ra_deg", "dec_deg", "tag"])
    writer.writeheader()
    return writer


def test_poll_radec_heartbeats_the_safety_watchdog_when_given_one():
    # Regression (safety): poll_radec had no safety parameter at all --
    # test_h_goto_characterization's own duration_s=5.0 call sits right at
    # SafetyGuard's default watchdog_timeout (5.0s), so any real-world
    # overhead (network round trips, scheduling jitter) risked the
    # watchdog's own backup :Q# firing mid-poll and corrupting exactly the
    # real GOTO this test exists to characterize. Now heartbeats every
    # iteration when a safety instance is passed.
    mock = MockMount(MockConfig())
    mount = Mount(mock)
    safety = _RecordingSafety()
    try:
        samples = poll_radec(mount, duration_s=0.3, hz=30.0, csv_writer=_csv_writer(), safety=safety)
        assert len(samples) > 0
        assert True in safety.calls
    finally:
        mount.close()


def test_poll_radec_works_without_a_safety_instance():
    # safety is optional -- every other poll_radec call site in
    # characterize.py stays comfortably under the watchdog timeout and
    # doesn't need this, so it must remain a no-op when omitted.
    mock = MockMount(MockConfig())
    mount = Mount(mock)
    try:
        samples = poll_radec(mount, duration_s=0.2, hz=30.0, csv_writer=_csv_writer())
        assert len(samples) > 0
    finally:
        mount.close()
