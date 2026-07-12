"""Best-effort system clock sync diagnostics.

The tracking loop's whole feedforward model depends on time.time() matching
true UTC (see am5/tracker.py's run_tracking_loop, which queries the
trajectory at now_wall + delta_t_s): a clock offset of even a few hundred
ms produces exactly the same symptom as a real mount response lag (a
stable along-track error), so this is a cheap thing to check/rule out
before assuming delta_t or mount_lag_s is the cause.

No new dependency: shells out to whichever of systemd-timesyncd
(timedatectl) or chrony (chronyc) is actually present, since which one is
installed varies by machine (this dev machine uses timesyncd; a
Raspberry Pi controlling the mount in the field commonly uses chrony).
Degrades gracefully to "unknown" if neither is available or parseable --
never raises.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ClockSyncStatus:
    synchronized: bool | None  # None if we couldn't determine even this much
    offset_s: float | None  # signed seconds (positive = system clock ahead of NTP), None if not measurable here
    source: str  # which tool produced this -- for the UI/debugging
    detail: str  # raw or human-readable text explaining the "why"


def check_clock_sync(timeout_s: float = 3.0) -> ClockSyncStatus:
    for probe in (_try_timedatectl_timesync, _try_chronyc, _try_timedatectl_status):
        status = probe(timeout_s)
        if status is not None:
            return status
    return ClockSyncStatus(
        synchronized=None, offset_s=None, source="none",
        detail="no clock-sync tool available (checked timedatectl, chronyc)",
    )


def _run(cmd: list[str], timeout_s: float) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


_TIMESYNC_UNIT_SECONDS = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0}
_TIMESYNC_OFFSET_RE = re.compile(r"Offset:\s*([+-]?[\d.]+)\s*(us|ms|s|min)")


def _try_timedatectl_timesync(timeout_s: float) -> ClockSyncStatus | None:
    """systemd-timesyncd's detailed view -- gives a real numeric offset,
    e.g. 'Offset: -6.451ms'. Only present on systemd 245+ with timesyncd
    actually active; absent (empty/error output) under chrony."""
    output = _run(["timedatectl", "timesync-status"], timeout_s)
    if not output:
        return None
    m = _TIMESYNC_OFFSET_RE.search(output)
    if m is None:
        return None
    offset_s = float(m.group(1)) * _TIMESYNC_UNIT_SECONDS[m.group(2)]
    return ClockSyncStatus(
        synchronized=abs(offset_s) < 1.0,  # a sanity threshold here, not an OS-reported flag
        offset_s=offset_s, source="timedatectl timesync-status", detail=output.strip(),
    )


_CHRONY_OFFSET_RE = re.compile(r"System time\s*:\s*([\d.]+)\s*seconds\s*(fast|slow)")


def _try_chronyc(timeout_s: float) -> ClockSyncStatus | None:
    """chrony's tracking view -- e.g. 'System time : 0.000012345 seconds
    fast of NTP time'. Common on Raspberry Pi / minimal server installs."""
    output = _run(["chronyc", "tracking"], timeout_s)
    if not output:
        return None
    m = _CHRONY_OFFSET_RE.search(output)
    if m is None:
        return ClockSyncStatus(synchronized=None, offset_s=None, source="chronyc tracking", detail=output.strip())
    magnitude, direction = float(m.group(1)), m.group(2)
    offset_s = magnitude if direction == "fast" else -magnitude
    return ClockSyncStatus(
        synchronized=abs(offset_s) < 1.0, offset_s=offset_s, source="chronyc tracking", detail=output.strip(),
    )


def _try_timedatectl_status(timeout_s: float) -> ClockSyncStatus | None:
    """Last resort: the generic status line every systemd machine has,
    regardless of which NTP client is actually doing the syncing --
    boolean only, no numeric offset."""
    output = _run(["timedatectl", "status"], timeout_s)
    if not output:
        return None
    synced = "System clock synchronized: yes" in output
    return ClockSyncStatus(synchronized=synced, offset_s=None, source="timedatectl status", detail=output.strip())
