from unittest.mock import patch

from am5.clock_sync import ClockSyncStatus, check_clock_sync


def _fake_run(stdout: str = "", returncode: int = 0):
    class _Result:
        pass

    r = _Result()
    r.stdout = stdout
    r.returncode = returncode
    return r


def test_check_clock_sync_never_raises_on_this_real_system():
    # No mocking -- exercises whatever tool this actual machine has, the
    # same "degrade gracefully, never crash" guarantee the module promises.
    status = check_clock_sync()
    assert isinstance(status, ClockSyncStatus)
    assert status.source != ""


def test_parses_timedatectl_timesync_status_offset_ms():
    output = "Server: ntp.ubuntu.com\n       Offset: -6.451ms\n        Delay: 48.369ms\n"
    with patch("subprocess.run", return_value=_fake_run(output)):
        status = check_clock_sync()
    assert status.source == "timedatectl timesync-status"
    assert status.offset_s == -0.006451
    assert status.synchronized is True


def test_parses_timedatectl_timesync_status_offset_seconds():
    output = "Offset: 1.5s\n"
    with patch("subprocess.run", return_value=_fake_run(output)):
        status = check_clock_sync()
    assert status.offset_s == 1.5
    assert status.synchronized is False  # >= 1.0s threshold


def test_falls_back_to_chronyc_when_timedatectl_timesync_unavailable():
    def fake_run(cmd, **_kwargs):
        if cmd[0] == "timedatectl" and cmd[1] == "timesync-status":
            return _fake_run("", returncode=1)
        if cmd[0] == "chronyc":
            return _fake_run("System time     : 0.000123456 seconds fast of NTP time\n")
        raise AssertionError(f"unexpected command {cmd}")

    with patch("subprocess.run", side_effect=fake_run):
        status = check_clock_sync()
    assert status.source == "chronyc tracking"
    assert status.offset_s == 0.000123456
    assert status.synchronized is True


def test_falls_back_to_timedatectl_status_boolean_when_nothing_else_parses():
    def fake_run(cmd, **_kwargs):
        if cmd == ["timedatectl", "timesync-status"]:
            return _fake_run("", returncode=1)
        if cmd[0] == "chronyc":
            return _fake_run("", returncode=127)
        if cmd == ["timedatectl", "status"]:
            return _fake_run("System clock synchronized: yes\nNTP service: active\n")
        raise AssertionError(f"unexpected command {cmd}")

    with patch("subprocess.run", side_effect=fake_run):
        status = check_clock_sync()
    assert status.source == "timedatectl status"
    assert status.synchronized is True
    assert status.offset_s is None


def test_returns_unknown_when_no_tool_is_available_at_all():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        status = check_clock_sync()
    assert status.source == "none"
    assert status.synchronized is None
    assert status.offset_s is None


def test_never_raises_on_a_timeout():
    import subprocess

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="timedatectl", timeout=3.0)):
        status = check_clock_sync()
    assert status.source == "none"
