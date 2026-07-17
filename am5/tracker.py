"""Feedforward rate-tracking loop. The only module that sends motion
commands to the mount for an actual pass (as opposed to characterize.py's
probing). Built on the empirical findings from characterize.py:

- :Rv is latched per-axis at :Me/:Mw/:Mn/:Ms time (test a/b).
- A manual rate ADDS to sidereal tracking — the net dRA/dt, dDEC/dt reported
  by :GMEQ# is exactly the commanded rate regardless of tracking state
  (test d) — so the commanded x-sidereal rate is just
  trajectory_rate_deg_s / SIDEREAL_DEG_PER_S, no separate sidereal term.
- Re-issuing :Rv+:Me every tick, same direction or reversed, is a smooth
  ramp with no stop/restart dead zone (test f/g) — so the loop can just
  push a new rate every tick without guarding direction changes.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .angles import angular_separation_deg, circular_diff_hours
from .constants import SIDEREAL_DEG_PER_S
from .ephemeris import Trajectory
from .logging_utils import utc_now_iso
from .mount import Mount
from .protocol import ProtocolError, parse_error
from .safety import SafetyGuard

# Limit-related :GAT# codes from the protocol table (brief section "Notes de
# planification"): these silently halt tracking without stopping motion
# commands from being accepted, so they must be polled for, not inferred.
LIMIT_ERROR_CODES = {5, 6, 8}


@dataclass
class AxisSigns:
    """Which physical direction increases RA/DEC on this mount session.
    Determined by wiring, not the protocol — recalibrate per session rather
    than trust a hardcoded value (characterize.py saw the DEC sign flip
    between runs near the pole -- root-caused later, see below).

    dec's sign is only valid for the pier side it was measured on. Root
    cause (confirmed on real AM3 hardware, not a guess): this is a German
    equatorial mount, and a pier flip physically rotates the DEC axis
    180 deg relative to the sky, while RA's axis (the polar axis) is
    unaffected. Measured directly: calibrating on pier side E gave
    dec=+1; immediately re-calibrating on side W (same session, same
    mount, only the pier side changed via a discrete :MS# GOTO) gave
    dec=-1, with ra unchanged at +1.

    update_pier_side() below is right for THAT case (a real :MS# GOTO
    landing on a different side). It is deliberately NOT wired into any
    automatic call site anymore (run_tracking_loop's periodic check,
    jog_goto's per-tick check, and the GUI's idle-poll handler all had it
    and all had it removed) after two real incidents: it fired during a
    plain jog_goto and during live ISS tracking -- neither involves a
    real :MS# GOTO, and in both cases the resulting DEC sign flip was
    WRONG, itself causing a real divergence (~35 deg during the tracking
    incident, correctly caught by the runaway guard, but the auto-flip is
    what caused it). Whether Mount.get_pier_side()'s :Gm# reading tracks
    true mechanical pier state during continuous motion (jog or tracking)
    the way it demonstrably does for a discrete GOTO is UNRESOLVED --
    real-hardware testing so far neither confirms nor rules out that
    :Gm# is "just" a computed hour-angle value that can read differently
    without the DEC motor's actual sense having changed at all. Until
    that's settled, the safe default is: recalibrate by hand
    (CalibrationPanel's "Calibrate axis directions") after any deliberate
    re-point, rather than trust an automatic correction that has twice
    caused the exact problem it was meant to prevent."""

    ra: float  # +1 if 'e' increases RA, -1 if 'e' decreases RA
    dec: float  # +1 if 'n' increases DEC, -1 if 'n' decreases DEC
    # 'E'/'W' pier side dec is currently correct for, or None if never
    # established (no calibration/observation yet -- see update_pier_side).
    calibrated_pier_side: str | None = None

    def update_pier_side(self, current_side: str | None) -> bool:
        """Correct for a real, discrete :MS# GOTO landing on a different
        pier side than dec was calibrated for (confirmed correct for that
        case). NOT currently called automatically anywhere -- see this
        class's own docstring for why (two real incidents from wiring it
        into continuous-motion loops). Available for a deliberate,
        explicit re-check after a real GOTO if a caller wants one.

        Call with a live :Gm# reading (Mount.get_pier_side()). Flips dec
        in place and returns True if `current_side` differs from the side
        dec was last known-correct
        for. 'N' (home/zero position -- no direction, see get_pier_side's
        docstring) and None/unknown are ignored: they don't tell us which
        side we'd actually be tracking from, so recording them would risk
        a false "flip" the next time a real E/W reading comes in."""
        if current_side not in ("E", "W"):
            return False
        if self.calibrated_pier_side is None:
            self.calibrated_pier_side = current_side
            return False
        if current_side != self.calibrated_pier_side:
            self.dec = -self.dec
            self.calibrated_pier_side = current_side
            return True
        return False


def calibrate_directions(
    mount: Mount, nudge_rate_x: float = 30.0, nudge_duration_s: float = 0.4,
    abort: threading.Event | None = None, safety: SafetyGuard | None = None,
) -> AxisSigns:
    """Small real nudge on each axis (~30x sidereal for ~0.4s, a few arcmin
    of travel) to empirically determine the sign convention for this
    session, rather than trusting a hardcoded assumption. `abort` (set by
    an emergency stop) short-circuits before the second-axis nudge so an
    e-stop mid-calibration isn't overridden by the next move command.

    `safety`, if given, is fed a movement_active heartbeat around each
    nudge -- am5/safety.py's own module docstring states this as a
    non-negotiable requirement for every entry point that moves the mount,
    but this function went without it (unlike every other motion-issuing
    path in this codebase, e.g. measure_mount_lag's matching parameter) --
    each nudge here is short (nudge_duration_s, well under a real
    watchdog_timeout) so the practical exposure window was narrow, but a
    hang between move() and stop() (e.g. time.sleep interrupted, the
    process wedged) would otherwise leave the mount jogging with zero
    safety-net coverage, unlike everywhere else.

    Records the current pier side as the one dec is now correct for (see
    AxisSigns.calibrated_pier_side/update_pier_side) -- calibrating doesn't
    itself change pier side, so whatever :Gm# reports right now is exactly
    the side this dec_sign was measured on."""
    ra0 = mount.get_radec()
    mount.set_rate(nudge_rate_x)
    mount.move("e")
    if safety is not None:
        safety.notify_command(movement_active=True)
    time.sleep(nudge_duration_s)
    mount.stop("e")
    if safety is not None:
        safety.notify_command(movement_active=False)
    time.sleep(0.1)
    ra1 = mount.get_radec()
    ra_sign = 1.0 if circular_diff_hours(ra1.ra_hours, ra0.ra_hours) > 0 else -1.0
    # 'N' (still at home) isn't a determinate side -- storing it as-is would
    # make update_pier_side() see a false "flip" (N != E/W) the moment the
    # mount later moves off home to a real side, even though pier side never
    # actually changed. None matches update_pier_side's own "first real E/W
    # reading just records, doesn't flip" handling.
    pier_side = _get_pier_side_safe(mount)
    if pier_side not in ("E", "W"):
        pier_side = None

    if abort is not None and abort.is_set():
        return AxisSigns(ra=ra_sign, dec=1.0, calibrated_pier_side=pier_side)

    dec0 = mount.get_radec()
    mount.set_rate(nudge_rate_x)
    mount.move("n")
    if safety is not None:
        safety.notify_command(movement_active=True)
    time.sleep(nudge_duration_s)
    mount.stop("n")
    if safety is not None:
        safety.notify_command(movement_active=False)
    time.sleep(0.1)
    dec1 = mount.get_radec()
    dec_sign = 1.0 if (dec1.dec_deg - dec0.dec_deg) > 0 else -1.0

    return AxisSigns(ra=ra_sign, dec=dec_sign, calibrated_pier_side=pier_side)


def _get_pier_side_safe(mount: Mount) -> str | None:
    try:
        return mount.get_pier_side()
    except ProtocolError:
        return None


@dataclass(frozen=True)
class MountLagResult:
    lag_s: float  # time from the :Rv#+:Me# command to ~settle_fraction of steady-state rate
    steady_rate_arcsec_s: float  # measured steady-state rate, to sanity-check against the commanded rate_x
    samples: int
    decel_lag_s: float = 0.0  # time from the stop command to velocity dropping below (1-settle_fraction)*steady_rate
    stop_command_t: float = 0.0  # offset (seconds since the move command) at which stop() was issued -- for plotting
    velocity_samples: tuple[tuple[float, float], ...] = ()  # (t_since_command, velocity_arcsec_s), accel then decel
    axis: str = "ra"  # which axis was actually stepped -- "ra" or "dec"


def _lag_result_from_samples(
    samples: list[tuple[float, float]], stop_command_t: float, decel_duration_s: float,
    settle_fraction: float, axis: str,
) -> MountLagResult:
    """Shared rise-time/fall-time analysis for one axis's (t_since_command,
    position_deg) samples -- see measure_mount_lag's docstring for the
    method. Factored out so both axes (measured simultaneously) get
    identical treatment."""
    if len(samples) < 4:
        return MountLagResult(lag_s=0.0, steady_rate_arcsec_s=0.0, samples=len(samples), axis=axis)

    velocities_arcsec_s = [
        (t_b, (p_b - p_a) * 3600.0 / (t_b - t_a))
        for (t_a, p_a), (t_b, p_b) in zip(samples, samples[1:])
        if t_b > t_a
    ]
    if not velocities_arcsec_s:
        return MountLagResult(lag_s=0.0, steady_rate_arcsec_s=0.0, samples=len(samples), axis=axis)

    accel_velocities = [(t, v) for t, v in velocities_arcsec_s if t <= stop_command_t]
    decel_velocities = [(t, v) for t, v in velocities_arcsec_s if t > stop_command_t]
    if not accel_velocities:
        accel_velocities = velocities_arcsec_s

    tail = accel_velocities[-max(1, len(accel_velocities) // 3):]
    steady_rate = sum(v for _, v in tail) / len(tail)

    threshold = abs(steady_rate) * settle_fraction
    lag_s = stop_command_t  # fallback: never clearly reached the threshold before the axis stopped
    for t, v in accel_velocities:
        if abs(v) >= threshold:
            lag_s = t
            break

    decel_threshold = abs(steady_rate) * (1.0 - settle_fraction)
    decel_lag_s = decel_duration_s  # fallback: never clearly settled within the decel window
    for t, v in decel_velocities:
        if abs(v) <= decel_threshold:
            decel_lag_s = t - stop_command_t
            break

    return MountLagResult(
        lag_s=lag_s, steady_rate_arcsec_s=steady_rate, samples=len(samples),
        decel_lag_s=decel_lag_s, stop_command_t=stop_command_t,
        velocity_samples=tuple(velocities_arcsec_s), axis=axis,
    )


def measure_mount_lag(
    mount: Mount, rate_x: float = 1440.0, duration_s: float = 2.5, poll_interval_s: float = 0.02,
    settle_fraction: float = 0.9, decel_duration_s: float | None = None,
    abort: threading.Event | None = None, safety: SafetyGuard | None = None,
) -> tuple[MountLagResult, MountLagResult]:
    """Empirically measures how long the mount takes to reach commanded
    angular rate after a step :Rv#+:Me#/:Mn#/:Ms# command -- a real motor
    doesn't reach the commanded rate instantaneously, and that ramp is a
    plausible contributor to the small, stable along-track lag seen on a
    real tracking run (confirmed present, ~20 arcsec at 20x/10x-sidereal
    rates, on real AM3 hardware -- see run_tracking_loop's module
    docstring; whether it's this ramp, serial round-trip latency, or a
    clock offset wasn't distinguished by that one test).

    Steps RA and DEC SIMULTANEOUSLY, returning (ra_result, dec_result).
    This costs nothing extra over stepping one axis: a single :GMEQ# poll
    already reports both ra_hours and dec_deg together, and "per_axis"
    :Rv latching (confirmed on real hardware, see test (a) in
    characterize.py) means each axis's commanded rate is independent of
    when the other axis's :Me#/:Mn# was sent -- so interleaving the two
    move commands measures the same real ramp as moving them one at a
    time (confirmed: DEC alone measured 1.33s/1.34s accel/decel and a
    4.56 deg/s^2 implied acceleration on real AM5 hardware, matching RA's
    1.32s/1.33s and ~4.5-4.6 deg/s^2 within measurement noise), while also
    matching how run_tracking_loop actually drives both axes together
    rather than one at a time.

    DEC jogs north or south, chosen dynamically from the CURRENT dec
    reading -- away from whichever pole is closer. That dynamic pick
    matters because manual jog rates aren't altitude-limit-checked the way
    a GOTO is (characterize.py test h's "e2" error is :MS#-only), and a few
    seconds at rate_x=1440 (~6 deg/s) covers several degrees -- a real risk
    of driving into a hard mechanical stop near the pole at speed if the
    direction were fixed.

    Polls :GMEQ# as fast as the real round trip allows (measured ~5ms by
    characterize.py, so poll_interval_s's real cadence is round-trip-bound,
    not this value), then finds when each axis's velocity first reaches
    settle_fraction of its own steady-state average (the last third of
    samples). This is a rise-time measurement, not a fitted transfer-
    function model -- it's meant to feed a single feedforward time-shift
    (TrackingConfig.mount_lag_s), not drive a real control model.

    Safety (added after a real audit finding): this is the only motion-
    commanding routine that polls :GMEQ# WHILE the axes are slewing, so
    it must protect against a mid-poll failure leaving the mount running.
    Two guards: (1) the whole move/poll is in a try/finally that ALWAYS
    stops both axes, even if the loop raises for any reason -- before this,
    a single ProtocolError from get_radec() propagated out and skipped the
    stop, leaving the mount slewing at rate_x indefinitely; (2) a transient
    ProtocolError on one poll is caught and the sample skipped rather than
    aborting the whole measurement (same resilience as run_tracking_loop's
    own poll). `safety`, if given, is fed a movement_active heartbeat each
    poll and reset to inactive on exit, so the SafetyGuard watchdog
    actually covers this slew -- run_tracking_loop and jog_goto both do
    this; this routine silently didn't, so the watchdog's backup :Q# was
    disabled for exactly the routine that most needed it.

    Real-hardware behavior, confirmed on real AM5 hardware (tube removed):
    at rate_x=1000, RA's lag_s~=0.92s and steady_rate_arcsec_s matched
    rate_x * SIDEREAL_ARCSEC_PER_S within 0.3% -- both the rise-time
    measurement and the rate sanity-check hold up against real motion. The
    reverse-engineered protocol notes (docs/AM5_UART_protocol_1.8.8.md,
    selector 5) independently confirm *why* it's not instantaneous: the
    firmware ramps its internal rate scalar by one integer per control
    update toward the requested value, both accelerating and decelerating
    -- so the deceleration phase below is expected to be a comparably
    gradual ramp, not a step, exactly like :Q#'s "pending stop" flag
    (selector 8) implies.

    After the accel phase, also commands a stop on both axes and keeps
    polling for `decel_duration_s` (defaults to `duration_s`, i.e. the same
    window) to capture the ramp-down the same way -- gives `decel_lag_s`
    (time from the stop command to velocity dropping back below
    (1-settle_fraction) of steady_rate) and lets a caller plot the full
    accel+decel speed curve from `velocity_samples` (e.g. the GUI's
    Diagnostics panel).
    """
    decel_duration_s = duration_s if decel_duration_s is None else decel_duration_s
    baseline_dec = mount.get_radec().dec_deg
    dec_dir = "s" if baseline_dec > 0 else "n"

    ra_samples: list[tuple[float, float]] = []  # (t_since_command, ra_deg), accel then decel
    dec_samples: list[tuple[float, float]] = []  # (t_since_command, dec_deg), accel then decel
    stop_command_t = duration_s  # fallback if aborted before the accel phase completes
    # Back on the shared set_rate() (:Rv#), not set_rate_ra/set_rate_dec
    # (:Rvr#/:Rvd#) -- :Rvr#/:Rvd# are undocumented in the official v1.7
    # PDF and, per a real-hardware re-test, their OWN readback (:GFR3#/
    # :GFD3#) showed real anomalies (RA consistently reading 1.00 below
    # what was just commanded; DEC's readback appeared frozen across a
    # later cross-clobber check) even though the actual commanded MOTION
    # matched correctly in every trial. Given that uncertainty (and that
    # ZWO's own forum reportedly says this isn't officially finished),
    # reverted to the thoroughly-proven shared approach here. Mount.
    # set_rate_ra/set_rate_dec and protocol.build_rv_ra/build_rv_dec are
    # kept fully implemented and tested (see test_mount.py's
    # test_set_rate_ra_and_dec_are_independent) -- swap the two lines
    # below back to them if :Rvr#/:Rvd# get more confidence later.
    mount.set_rate(rate_x)
    try:
        mount.move("e")
        mount.set_rate(rate_x)
        mount.move(dec_dir)
        command_t = time.monotonic()
        while time.monotonic() - command_t < duration_s:
            if abort is not None and abort.is_set():
                break
            if safety is not None:
                safety.notify_command(movement_active=True)  # heartbeat -- keeps the watchdog covering this slew
            try:
                radec = mount.get_radec()
            except ProtocolError:
                # A transient bad reply must not abort the measurement
                # (which would skip the stop in the finally is moot now,
                # but would still throw away the whole run) -- skip this
                # one sample and keep going.
                time.sleep(poll_interval_s)
                continue
            t = time.monotonic() - command_t
            ra_samples.append((t, radec.ra_hours * 15.0))
            dec_samples.append((t, radec.dec_deg))
            time.sleep(poll_interval_s)

        stop_command_t = time.monotonic() - command_t
        mount.stop("e")
        mount.stop(dec_dir)
        if safety is not None:
            safety.notify_command(movement_active=False)
        decel_start = time.monotonic()
        while time.monotonic() - decel_start < decel_duration_s:
            if abort is not None and abort.is_set():
                break
            try:
                radec = mount.get_radec()
            except ProtocolError:
                time.sleep(poll_interval_s)
                continue
            t = time.monotonic() - command_t
            ra_samples.append((t, radec.ra_hours * 15.0))
            dec_samples.append((t, radec.dec_deg))
            time.sleep(poll_interval_s)
    finally:
        # Always stop both axes, whatever happened above -- a mount left
        # slewing after a lag measurement is a real hazard (the incident
        # this guards). Harmless to call again if the decel phase above
        # already stopped them. Best-effort so a failing stop() can't mask
        # an in-flight exception, and the watchdog is released last.
        try:
            mount.stop("e")
        finally:
            try:
                mount.stop(dec_dir)
            finally:
                if safety is not None:
                    safety.notify_command(movement_active=False)

    ra_result = _lag_result_from_samples(ra_samples, stop_command_t, decel_duration_s, settle_fraction, axis="ra")
    dec_result = _lag_result_from_samples(dec_samples, stop_command_t, decel_duration_s, settle_fraction, axis="dec")
    return ra_result, dec_result


@dataclass
class LiveOffsets:
    """Operator-adjustable corrections, live during a pass. Thread-safe:
    written by am5.live_input's keyboard thread, read by the tracking loop.

    delta_t_s shifts which point of the precomputed trajectory we track —
    persistent, drifts slowly as the operator dials it in.

    The perpendicular nudge is a bounded-duration rate pulse (like a
    hand-controller tap), not a persistent position offset: a keyboard has
    no reliable key-held signal, and a timed pulse sidesteps having to turn
    a discrete position jump into an instantaneous rate.
    """

    delta_t_s: float = 0.0
    _perp_sign: float = field(default=0.0, repr=False)
    _perp_until: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def adjust_delta_t(self, delta: float) -> None:
        with self._lock:
            self.delta_t_s += delta

    def reset(self) -> None:
        """Clears delta_t_s and any in-flight perp pulse -- call this
        whenever a NEW pass is selected (see TransitPanel.set_trajectory).

        Regression fix: delta_t_s used to persist for the whole app
        session with no reset anywhere -- a clock-offset/along-track
        correction dialed in by hand during one pass silently carried
        over and got applied to the very first tick of the NEXT pass's
        trajectory query (run_tracking_loop's t_query = now + delta_t_s +
        mount_lag_s). The ISS moves at roughly 900-1400 arcsec/s during a
        typical pass (measured against a real trajectory), so even a
        leftover 0.5s of stale delta_t already produces a ~7.5 arcmin
        along-track offset at the new pass's start -- larger than a
        typical narrow-FOV main camera's entire field -- with nothing
        warning the operator beyond a small, easy-to-miss "+X.Xs" label."""
        with self._lock:
            self.delta_t_s = 0.0
            self._perp_sign = 0.0
            self._perp_until = 0.0

    def trigger_perp_pulse(self, sign: float, duration_s: float = 0.15) -> None:
        with self._lock:
            self._perp_sign = sign
            self._perp_until = time.monotonic() + duration_s

    def snapshot(self) -> tuple[float, float]:
        """(delta_t_s, perp_sign) — perp_sign is 0.0 outside an active pulse."""
        with self._lock:
            active = time.monotonic() < self._perp_until
            return self.delta_t_s, (self._perp_sign if active else 0.0)


# Angular rate of a perpendicular nudge pulse, in real sky arcsec/s — a gentle
# correction, not a slew (~20x sidereal).
PERP_NUDGE_DEG_S = 20.0 * SIDEREAL_DEG_PER_S


def _perp_rate_components(dec_deg: float, dra_dt_deg_s: float, ddec_dt_deg_s: float, sign: float) -> tuple[float, float]:
    """Extra (dra_dt, ddec_dt) in raw RA-degrees/s and DEC-degrees/s to add
    for a perpendicular-to-track nudge of the given sign. Perpendicular is
    defined in the tangent plane (RA scaled by cos(dec)) so it's a true
    sky-perpendicular, not just "swap RA and DEC"."""
    if sign == 0.0:
        return 0.0, 0.0
    cos_dec = math.cos(math.radians(dec_deg))
    v_ra_tan = dra_dt_deg_s * cos_dec
    v_dec = ddec_dt_deg_s
    speed = math.hypot(v_ra_tan, v_dec)
    if speed < 1e-9:
        return 0.0, sign * PERP_NUDGE_DEG_S  # no track direction yet (e.g. at rest) — just nudge DEC
    perp_ra_tan = -v_dec / speed * sign
    perp_dec = v_ra_tan / speed * sign
    extra_dra_dt = (perp_ra_tan * PERP_NUDGE_DEG_S) / cos_dec if cos_dec > 1e-6 else 0.0
    extra_ddec_dt = perp_dec * PERP_NUDGE_DEG_S
    return extra_dra_dt, extra_ddec_dt


def decompose_error(
    d_ra_deg: float, d_dec_deg: float, dec_deg: float, dra_dt_deg_s: float, ddec_dt_deg_s: float
) -> tuple[float, float]:
    """Split a (d_ra_deg, d_dec_deg) position error into (along_track_deg,
    cross_track_deg) relative to the instantaneous velocity direction, in
    the tangent plane. Originally just logged for the operator (an
    along-track error can come from a clock offset -- delta_t would fix
    that -- or from the mount's own rate-change lag -- delta_t makes that
    worse, not better -- and those two cases aren't distinguishable from a
    single error sample, so along-track correction stays manual). Also used
    by camera/guiding.py to decompose a camera-detected pixel offset the
    same way, since cross-track error (unlike along-track) has no such
    ambiguity and is safe to auto-correct."""
    cos_dec = math.cos(math.radians(dec_deg))
    err_ra_tan = d_ra_deg * cos_dec
    err_dec = d_dec_deg
    v_ra_tan = dra_dt_deg_s * cos_dec
    v_dec = ddec_dt_deg_s
    speed = math.hypot(v_ra_tan, v_dec)
    if speed < 1e-9:
        return 0.0, math.hypot(err_ra_tan, err_dec)
    along = (err_ra_tan * v_ra_tan + err_dec * v_dec) / speed
    cross = (err_ra_tan * -v_dec + err_dec * v_ra_tan) / speed
    return along, cross


def _along_cross_rate_to_equatorial(
    along_rate_deg_s: float, cross_rate_deg_s: float, dec_deg: float,
    dra_dt_deg_s: float, ddec_dt_deg_s: float, max_deg_s: float,
) -> tuple[float, float]:
    """Inverse of decompose_error's projection: turns an (along, cross)
    rate correction back into raw (extra_dra_dt, extra_ddec_dt) -- used by
    run_tracking_loop's optional feedback trim (TrackingConfig.
    enable_feedback). Same tangent-plane basis as decompose_error/
    _perp_rate_components (velocity direction = along, its
    tangent-plane-perpendicular = cross), and clamped to max_deg_s
    (combined along+cross magnitude) so a feedback term can never dominate
    the feedforward rate it's trimming."""
    cos_dec = math.cos(math.radians(dec_deg))
    v_ra_tan = dra_dt_deg_s * cos_dec
    v_dec = ddec_dt_deg_s
    speed = math.hypot(v_ra_tan, v_dec)
    if speed < 1e-9:
        # no track direction to project onto -- treat "along" as RA-tangent, "cross" as DEC, arbitrarily
        ra_tan, dec_comp = along_rate_deg_s, cross_rate_deg_s
    else:
        unit_along_ra_tan, unit_along_dec = v_ra_tan / speed, v_dec / speed
        unit_cross_ra_tan, unit_cross_dec = -v_dec / speed, v_ra_tan / speed
        ra_tan = along_rate_deg_s * unit_along_ra_tan + cross_rate_deg_s * unit_cross_ra_tan
        dec_comp = along_rate_deg_s * unit_along_dec + cross_rate_deg_s * unit_cross_dec

    magnitude = math.hypot(ra_tan, dec_comp)
    if magnitude > max_deg_s and magnitude > 1e-12:
        scale = max_deg_s / magnitude
        ra_tan *= scale
        dec_comp *= scale

    extra_dra_dt = ra_tan / cos_dec if cos_dec > 1e-6 else 0.0
    return extra_dra_dt, dec_comp


def _pick_direction(signed_rate: float, sign_convention: float, positive_dir: str, negative_dir: str) -> str:
    wants_positive = signed_rate >= 0.0
    convention_positive = sign_convention > 0.0
    return positive_dir if (wants_positive == convention_positive) else negative_dir


@dataclass
class TrackingConfig:
    loop_hz: float = 20.0
    max_rate_x: float = 1400.0  # safety margin under the measured 1440x max
    error_log_hz: float = 1.0  # cadence for the along/cross-track error printout
    status_check_hz: float = 1.0
    # Auto-stop if the measured pointing error ever exceeds this, meaning
    # the mount is running AWAY from the target rather than following it --
    # the signature of a wrong axis-sign calibration, a wrong rate model, or
    # a stall on an untested setup. Set generously (10 deg): the ISS never
    # needs >~440x sidereal (its DEC maxes at the 51.6 deg orbital
    # inclination, so RA rates don't blow up like near the pole), so
    # feedforward lag stays small even on a sluggish mount, while a true
    # wrong-sign runaway diverges to tens of degrees within seconds. Set to
    # 0 to disable.
    runaway_stop_deg: float = 10.0

    # From measure_mount_lag() -- shifts the feedforward query time earlier
    # by this much, so the trajectory is queried for "where the target will
    # be by the time the mount actually gets there" instead of "now".
    # Defaults to 0.0 (no change from prior behavior) until measured. Used
    # as-is (a flat per-tick shift) when max_accel_deg_s2 is None; otherwise
    # treated as an upper bound -- see max_accel_deg_s2 below.
    mount_lag_s: float = 0.0

    # Optional refinement on top of mount_lag_s, also from measure_mount_lag()
    # (steady_rate_arcsec_s / lag_s, roughly -- the GUI's Diagnostics panel
    # fills this in automatically alongside mount_lag_s). mount_lag_s alone
    # was measured from the single biggest possible rate change (a 0-to-max
    # step at pass acquisition) and applied identically on every tick,
    # including mid-pass ticks where the ISS's rate barely changes from one
    # tick to the next -- a real risk of overcorrecting the ticks that need
    # it least (characterize.py's relatch-smoothness test showed small
    # mid-slew rate changes settle much faster than a full 0-to-max ramp).
    # When set, run_tracking_loop instead estimates how big THIS tick's
    # rate change actually is and leads by only as much time as that change
    # would take to physically ramp at this rate, capped at mount_lag_s --
    # so a big acquisition-time jump still gets the full lead, while a
    # small mid-pass adjustment gets a proportionally smaller one. None
    # (default) keeps the old flat-mount_lag_s behavior unchanged.
    max_accel_deg_s2: float | None = None

    # Closed-loop trim on top of the feedforward rate -- opt-in (default
    # off) and unvalidated on real hardware as of this writing. A slow
    # (same ~1Hz cadence as the error-log poll, not per-tick) PI correction
    # on the along/cross-track position error, so it doesn't care WHETHER
    # the residual error is a clock offset, mount ramp lag, or anything
    # else (see decompose_error's docstring on why delta_t deliberately
    # can't auto-correct along-track: this sidesteps that ambiguity instead
    # of resolving it, since integral control drives steady-state error to
    # zero regardless of cause). Defaults are deliberately gentle -- this
    # is a trim, not a replacement for a reasonable feedforward model.
    enable_feedback: bool = False
    feedback_kp: float = 0.1  # (deg/s correction) per (deg of error), roughly 1/s
    feedback_ki: float = 0.002  # (deg/s correction) per (deg*s of accumulated error)
    feedback_max_correction_deg_s: float = 0.0014  # hard clamp, ~5 arcsec/s -- can't dominate the feedforward term
    feedback_integral_limit_deg: float = 0.05  # anti-windup clamp on the accumulated error (~180 arcsec)

    # If not None, configured on the mount before every pass (:ST#, see
    # Mount.set_meridian_behavior) instead of relying on whatever the
    # mount's own factory/current default happens to be -- root cause of a
    # real "tracking diverges badly right after the meridian" incident:
    # the protocol doc's own default (track_past_meridian=False) silently
    # stops RA tracking just 1 degree past the meridian while this loop
    # keeps sending rate commands, unaware. 15 deg is the protocol's own
    # maximum allowance -- generous for a ~6min ISS pass.
    #
    # DEFAULTS TO None (disabled) DELIBERATELY, even though it fixes a
    # real bug: Mount.set_meridian_behavior's reply format is UNCONFIRMED
    # against real hardware (see its own docstring) -- if the mount
    # actually replies "1#" rather than the assumed bare "1", the
    # leftover '#' byte stays sitting in the serial buffer and prefixes
    # the NEXT command's reply, desyncing every subsequent read for the
    # rest of the session. That failure mode is WORSE than the meridian
    # bug this exists to fix (one bad pass vs. a corrupted session), so
    # this must not run unattended before being verified once on real
    # hardware. To verify: call mount.set_meridian_behavior(True, 15.0)
    # once by hand, then mount.get_meridian_behavior() immediately after
    # -- if that raises ProtocolError or returns garbage instead of
    # (False, True, 15.0), the reply format assumption was wrong and this
    # must not be enabled. Once confirmed, set this to e.g. 15.0 (or pass
    # an explicit TrackingConfig(meridian_track_limit_deg=15.0) per call).
    meridian_track_limit_deg: float | None = None


class TrackingRunaway(RuntimeError):
    """Raised by run_tracking_loop when pointing error exceeds
    TrackingConfig.runaway_stop_deg -- the mount is diverging, not
    following. run_tracking_loop stops the mount before raising."""


# Below this, build_rv's ":Rv%.2f#" formatting rounds to "0.00" -- see
# run_tracking_loop's own comment on why a rate this small is sent as
# stop(dir) instead of set_rate(rate)+move(dir).
MIN_COMMANDABLE_RATE_X = 0.005


def run_tracking_loop(
    mount: Mount,
    safety: SafetyGuard,
    trajectory: Trajectory,
    axis_signs: AxisSigns,
    offsets: LiveOffsets,
    csv_writer,
    duration_s: float,
    config: TrackingConfig | None = None,
    stop_event: threading.Event | None = None,
    on_tick: Callable[[dict], None] | None = None,
    on_limit_warning: Callable[[int], None] | None = None,
) -> None:
    cfg = config or TrackingConfig()
    period = 1.0 / cfg.loop_hz
    error_log_every = max(1, round(cfg.loop_hz / cfg.error_log_hz))
    status_every = max(1, round(cfg.loop_hz / cfg.status_check_hz))

    # Establish sidereal tracking before applying feedforward rates on top.
    # The feedforward model (module docstring) is "commanded :Me/:Mn rate
    # ADDS to sidereal, and reported dRA/dt == commanded rate" -- that model
    # is only unambiguous with the sidereal drive actually running. Enabling
    # it here makes the loop self-contained (doesn't rely on the operator
    # having toggled tracking first) and matches the conventional
    # guide-on-top-of-tracking workflow. Harmless if already on.
    try:
        mount.set_tracking(True)
    except ProtocolError as exc:
        print(f"[warn] could not enable sidereal tracking (:Te#) before pass: {exc}", file=sys.stderr)

    # See TrackingConfig.meridian_track_limit_deg's docstring -- without
    # this, a pass that runs long enough to cross the meridian can silently
    # stop on the mount side while this loop keeps commanding rates,
    # producing unbounded along/cross-track error that looks like a
    # tracking bug but is actually the mount having stopped listening.
    if cfg.meridian_track_limit_deg is not None:
        try:
            mount.set_meridian_behavior(track_past_meridian=True, limit_deg=cfg.meridian_track_limit_deg)
        except ProtocolError as exc:
            print(f"[warn] could not configure meridian tracking behavior (:ST#) before pass: {exc}", file=sys.stderr)

    last_limit_code: int | None = None
    t_loop_start = time.monotonic()
    next_tick = t_loop_start
    tick = 0
    # Feedback trim state (see TrackingConfig.enable_feedback) -- updated
    # at the same ~1Hz cadence as the error-log poll below (that's the only
    # place a fresh position measurement exists), held constant and applied
    # every tick in between.
    feedback_dra_dt = 0.0
    feedback_ddec_dt = 0.0
    integral_along_deg = 0.0
    integral_cross_deg = 0.0
    last_feedback_t = t_loop_start
    # Previous tick's commanded rate (x sidereal) -- only used when
    # max_accel_deg_s2 is set, to size this tick's lead time off how big a
    # rate CHANGE it's actually asking for (see TrackingConfig.max_accel_
    # deg_s2's docstring). Starts at 0: tracking typically begins from rest
    # or plain sidereal, so the first tick's jump to the ISS's rate is
    # correctly treated as the big acquisition-time transient it is.
    prev_ra_rate_x = 0.0
    prev_dec_rate_x = 0.0
    while (stop_event is None or not stop_event.is_set()) and time.monotonic() - t_loop_start < duration_s:
        now_wall = time.time()
        delta_t_s, perp_sign = offsets.snapshot()

        lag_s = cfg.mount_lag_s
        if cfg.max_accel_deg_s2:
            # Probe at the full (worst-case) lag first, purely to see how
            # large a rate change that would imply versus last tick's
            # commanded rate -- a small mid-pass adjustment ramps up in a
            # fraction of the full acquisition-time settle, so leading by
            # the full mount_lag_s would overcorrect it.
            probe_ra_deg, probe_dec_deg, probe_dra_dt, probe_ddec_dt = trajectory.interpolate(
                now_wall + delta_t_s + cfg.mount_lag_s
            )
            probe_ra_rate_x = probe_dra_dt / SIDEREAL_DEG_PER_S
            probe_dec_rate_x = probe_ddec_dt / SIDEREAL_DEG_PER_S
            delta_ra_deg_s = abs(probe_ra_rate_x - prev_ra_rate_x) * SIDEREAL_DEG_PER_S
            delta_dec_deg_s = abs(probe_dec_rate_x - prev_dec_rate_x) * SIDEREAL_DEG_PER_S
            needed_lag_s = max(delta_ra_deg_s, delta_dec_deg_s) / cfg.max_accel_deg_s2
            lag_s = min(cfg.mount_lag_s, needed_lag_s)

        t_query = now_wall + delta_t_s + lag_s

        ra_deg, dec_deg, dra_dt, ddec_dt = trajectory.interpolate(t_query)
        extra_dra, extra_ddec = _perp_rate_components(dec_deg, dra_dt, ddec_dt, perp_sign)
        dra_dt += extra_dra
        ddec_dt += extra_ddec
        if cfg.enable_feedback:
            dra_dt += feedback_dra_dt
            ddec_dt += feedback_ddec_dt

        ra_rate_x = dra_dt / SIDEREAL_DEG_PER_S
        dec_rate_x = ddec_dt / SIDEREAL_DEG_PER_S
        ra_rate_x_clamped = math.copysign(min(abs(ra_rate_x), cfg.max_rate_x), ra_rate_x) if ra_rate_x else 0.0
        dec_rate_x_clamped = math.copysign(min(abs(dec_rate_x), cfg.max_rate_x), dec_rate_x) if dec_rate_x else 0.0
        prev_ra_rate_x, prev_dec_rate_x = ra_rate_x_clamped, dec_rate_x_clamped

        ra_dir = _pick_direction(ra_rate_x_clamped, axis_signs.ra, "e", "w")
        dec_dir = _pick_direction(dec_rate_x_clamped, axis_signs.dec, "n", "s")
        # Regression fix: a feedforward rate that's (or rounds to, under
        # build_rv's %.2f formatting) zero used to still go through
        # set_rate(0)+move(dir) -- the reverse-engineered protocol notes
        # (docs/AM5_UART_protocol_1.8.8.md, GFR/GFD selector 5) say the
        # firmware's internal rate scalar floors fractional/very-low rates
        # at a minimum of 1x sidereal, meaning ":Rv0.00#"+":Me#" may not
        # actually hold the axis still -- it could creep at ~1x in
        # whatever direction _pick_direction picked for a zero-magnitude
        # rate. Unconfirmed on real hardware (never tested at near-zero
        # rates), but stop(dir) costs nothing and removes the risk either way.
        # Back on the shared set_rate() (:Rv#) -- see measure_mount_lag's
        # matching comment for why set_rate_ra/set_rate_dec (:Rvr#/:Rvd#)
        # were reverted here despite being fully implemented and tested.
        if abs(ra_rate_x_clamped) < MIN_COMMANDABLE_RATE_X:
            mount.stop(ra_dir)
        else:
            mount.set_rate(abs(ra_rate_x_clamped))
            mount.move(ra_dir)
        if abs(dec_rate_x_clamped) < MIN_COMMANDABLE_RATE_X:
            mount.stop(dec_dir)
        else:
            mount.set_rate(abs(dec_rate_x_clamped))
            mount.move(dec_dir)
        safety.notify_command(movement_active=True)

        actual_ra_deg = actual_dec_deg = ""
        if tick % error_log_every == 0:
            try:
                actual = mount.get_radec()
                actual_ra_deg = actual.ra_hours * 15.0
                actual_dec_deg = actual.dec_deg
                d_ra = circular_diff_hours(actual.ra_hours, ra_deg / 15.0) * 15.0
                d_dec = actual_dec_deg - dec_deg
                # Real great-circle separation, not a tangent-plane
                # hypot(d_ra*cos(dec), d_dec) approximation -- that
                # approximation is only valid for small separations and
                # actively misleads beyond a few degrees (confirmed on
                # real hardware for jog_goto's own divergence guard, see
                # angular_separation_deg's docstring; this runaway check
                # is the same class of bug and matters most exactly when
                # it's needed -- a real, large divergence, e.g. starting
                # a tracking pass late without first slewing onto target).
                total_error_deg = angular_separation_deg(actual.ra_hours * 15.0, actual_dec_deg, ra_deg, dec_deg)
                if cfg.runaway_stop_deg > 0 and total_error_deg > cfg.runaway_stop_deg:
                    mount.stop()
                    raise TrackingRunaway(
                        f"pointing error {total_error_deg:.1f} deg exceeds runaway limit "
                        f"{cfg.runaway_stop_deg} deg -- mount is diverging, not following. "
                        f"Check axis-sign calibration (did you run Calibrate?), the "
                        f"tracking rate model, and for a stall."
                    )
                along_deg, cross_deg = decompose_error(d_ra, d_dec, dec_deg, dra_dt, ddec_dt)

                feedback_note = ""
                if cfg.enable_feedback:
                    now_mono = time.monotonic()
                    dt_feedback = now_mono - last_feedback_t
                    last_feedback_t = now_mono
                    # Outside the trajectory's own active window -- an
                    # early start still "sitting at the boundary" (see
                    # Trajectory.interpolate's own docstring; dra_dt/
                    # ddec_dt are explicitly zeroed there) -- decompose_
                    # error's zero-speed branch returns a signless
                    # MAGNITUDE for cross_deg (always >= 0, confirmed by
                    # feeding it the same error with the sign flipped and
                    # getting an identical result back), not a real
                    # cross-track error. Integrating that every tick would
                    # windup a spurious, wrongly-directed correction while
                    # just waiting for the pass to start -- the same class
                    # of bug fixed on the auto-guide/finder-correction
                    # paths (am5/gui/panels.py). Freeze the feedback state
                    # instead of corrupting it with a fabricated direction;
                    # it resumes cleanly once real motion starts.
                    cos_dec_fb = math.cos(math.radians(dec_deg))
                    if math.hypot(dra_dt * cos_dec_fb, ddec_dt) >= 1e-9:
                        # decompose_error's along_deg/cross_deg are (actual -
                        # target), so the correction has to point the other
                        # way to close the gap -- error_* below flips to the
                        # standard PID convention (setpoint - measured) so
                        # positive gains do the intuitive thing.
                        error_along_deg = -along_deg
                        error_cross_deg = -cross_deg
                        limit = cfg.feedback_integral_limit_deg
                        integral_along_deg = max(-limit, min(limit, integral_along_deg + error_along_deg * dt_feedback))
                        integral_cross_deg = max(-limit, min(limit, integral_cross_deg + error_cross_deg * dt_feedback))
                        correction_along = cfg.feedback_kp * error_along_deg + cfg.feedback_ki * integral_along_deg
                        correction_cross = cfg.feedback_kp * error_cross_deg + cfg.feedback_ki * integral_cross_deg
                        feedback_dra_dt, feedback_ddec_dt = _along_cross_rate_to_equatorial(
                            correction_along, correction_cross, dec_deg, dra_dt, ddec_dt,
                            cfg.feedback_max_correction_deg_s,
                        )
                        feedback_note = (
                            f" | feedback trim: along {correction_along * 3600:+.2f}\"/s "
                            f"cross {correction_cross * 3600:+.2f}\"/s"
                        )

                print(f"[track] cross-track error: {cross_deg * 3600:+.1f}\" "
                      f"along-track: {along_deg * 3600:+.1f}\" — nudge delta_t/perp by hand if this grows"
                      f"{feedback_note}",
                      file=sys.stderr)
                if on_tick is not None:
                    on_tick({
                        "elapsed_s": time.monotonic() - t_loop_start,
                        "target_ra_deg": ra_deg % 360.0, "target_dec_deg": dec_deg,
                        "actual_ra_deg": actual_ra_deg, "actual_dec_deg": actual_dec_deg,
                        "along_track_arcsec": along_deg * 3600, "cross_track_arcsec": cross_deg * 3600,
                        "ra_rate_x": ra_rate_x_clamped, "dec_rate_x": dec_rate_x_clamped,
                        "delta_t_s": delta_t_s,
                        "feedback_dra_dt_arcsec_s": feedback_dra_dt * 3600 if cfg.enable_feedback else "",
                        "feedback_ddec_dt_arcsec_s": feedback_ddec_dt * 3600 if cfg.enable_feedback else "",
                    })
            except ProtocolError as exc:
                print(f"[warn] bad :GMEQ# reply during error-log poll: {exc}", file=sys.stderr)

        if tick % status_every == 0:
            limit_code = _check_limits(mount)
            # Only fire on a NEW code (first occurrence, or a change to a
            # different one) -- status_check_hz keeps polling every second
            # regardless, so without this a stuck limit would re-notify
            # once per second for the rest of the pass.
            if limit_code is not None and limit_code != last_limit_code and on_limit_warning is not None:
                on_limit_warning(limit_code)
            last_limit_code = limit_code

        csv_writer.writerow({
            "t_mono": time.monotonic(), "t_utc": utc_now_iso(),
            "target_ra_deg": ra_deg % 360.0, "target_dec_deg": dec_deg,
            "actual_ra_deg": actual_ra_deg, "actual_dec_deg": actual_dec_deg,
            "ra_rate_x": ra_rate_x_clamped, "dec_rate_x": dec_rate_x_clamped,
            "delta_t_s": delta_t_s, "perp_pulse": perp_sign,
        })

        tick += 1
        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


LIMIT_CODE_MEANINGS = {5: "below horizon", 6: "below altitude limit", 8: "meridian crossed"}


def _check_limits(mount: Mount) -> int | None:
    # Used to also auto-correct axis_signs.dec here from a live :Gm#
    # pier-side read -- tried and reverted, see AxisSigns' docstring in
    # this file for the full account. A real incident: this fired during
    # live ISS tracking and the resulting DEC sign flip caused a genuine
    # ~35 deg divergence (correctly caught by the runaway guard below,
    # but the flip is what caused it, not a real calibration or mount
    # problem). :Gm#'s relationship to true mechanical pier state during
    # continuous tracking (as opposed to a discrete :MS# GOTO, confirmed
    # correct) is unresolved.
    #
    # Returns the :GAT# error code if it's one of LIMIT_ERROR_CODES, else
    # None -- run_tracking_loop's own call site is what actually surfaces
    # this (both to stderr and, via on_limit_warning, to the GUI's log),
    # this function just polls and classifies.
    try:
        raw = mount.get_tracking_status()
    except ProtocolError as exc:
        print(f"[warn] bad :GAT# reply during status check: {exc}", file=sys.stderr)
        return None
    code = parse_error(raw)
    if code in LIMIT_ERROR_CODES:
        print(f"[SAFETY] :GAT# reports error {code} ({LIMIT_CODE_MEANINGS.get(code, '?')}) "
              f"— tracking may have silently stopped on the mount side", file=sys.stderr)
        return code
    return None


TRACKING_CSV_FIELDS = [
    "t_mono", "t_utc", "target_ra_deg", "target_dec_deg", "actual_ra_deg", "actual_dec_deg",
    "ra_rate_x", "dec_rate_x", "delta_t_s", "perp_pulse",
]
