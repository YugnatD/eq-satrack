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
    abort: threading.Event | None = None,
) -> AxisSigns:
    """Small real nudge on each axis (~30x sidereal for ~0.4s, a few arcmin
    of travel) to empirically determine the sign convention for this
    session, rather than trusting a hardcoded assumption. `abort` (set by
    an emergency stop) short-circuits before the second-axis nudge so an
    e-stop mid-calibration isn't overridden by the next move command.

    Records the current pier side as the one dec is now correct for (see
    AxisSigns.calibrated_pier_side/update_pier_side) -- calibrating doesn't
    itself change pier side, so whatever :Gm# reports right now is exactly
    the side this dec_sign was measured on."""
    ra0 = mount.get_radec()
    mount.set_rate(nudge_rate_x)
    mount.move("e")
    time.sleep(nudge_duration_s)
    mount.stop("e")
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
    time.sleep(nudge_duration_s)
    mount.stop("n")
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


def measure_mount_lag(
    mount: Mount, rate_x: float = 100.0, duration_s: float = 1.5, poll_interval_s: float = 0.02,
    settle_fraction: float = 0.9, abort: threading.Event | None = None,
) -> MountLagResult:
    """Empirically measures how long the mount takes to reach commanded
    angular rate after a step :Rv#+:Me# command -- a real motor doesn't
    reach the commanded rate instantaneously, and that ramp is a plausible
    contributor to the small, stable along-track lag seen on a real
    tracking run (confirmed present, ~20 arcsec at 20x/10x-sidereal rates,
    on real AM3 hardware -- see run_tracking_loop's module docstring;
    whether it's this ramp, serial round-trip latency, or a clock offset
    wasn't distinguished by that one test).

    Steps the RA axis (arbitrary choice -- DEC is mechanically identical
    on this mount) at rate_x for duration_s, polling :GMEQ# as fast as the
    real round trip allows (measured ~5ms by characterize.py, so
    poll_interval_s's real cadence is round-trip-bound, not this value),
    then finds when velocity first reaches settle_fraction of its own
    steady-state average (the last third of samples). This is a rise-time
    measurement, not a fitted transfer-function model -- it's meant to
    feed a single feedforward time-shift (TrackingConfig.mount_lag_s), not
    drive a real control model.

    Real-hardware behavior as of this writing: unvalidated. Written and
    tested against MockMount only -- run it for real and sanity-check
    steady_rate_arcsec_s against rate_x * SIDEREAL_ARCSEC_PER_S before
    trusting lag_s for anything.
    """
    samples: list[tuple[float, float]] = []  # (t_since_command, ra_hours)
    mount.set_rate(rate_x)
    mount.move("e")
    command_t = time.monotonic()
    while time.monotonic() - command_t < duration_s:
        if abort is not None and abort.is_set():
            break
        radec = mount.get_radec()
        samples.append((time.monotonic() - command_t, radec.ra_hours))
        time.sleep(poll_interval_s)
    mount.stop("e")

    if len(samples) < 4:
        return MountLagResult(lag_s=0.0, steady_rate_arcsec_s=0.0, samples=len(samples))

    velocities_arcsec_s = [
        (t_b, (ra_b - ra_a) * 15.0 * 3600.0 / (t_b - t_a))
        for (t_a, ra_a), (t_b, ra_b) in zip(samples, samples[1:])
        if t_b > t_a
    ]
    if not velocities_arcsec_s:
        return MountLagResult(lag_s=0.0, steady_rate_arcsec_s=0.0, samples=len(samples))

    tail = velocities_arcsec_s[-max(1, len(velocities_arcsec_s) // 3):]
    steady_rate = sum(v for _, v in tail) / len(tail)

    threshold = abs(steady_rate) * settle_fraction
    lag_s = duration_s  # fallback: never clearly reached the threshold within the test window
    for t, v in velocities_arcsec_s:
        if abs(v) >= threshold:
            lag_s = t
            break

    return MountLagResult(lag_s=lag_s, steady_rate_arcsec_s=steady_rate, samples=len(samples))


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
    # Defaults to 0.0 (no change from prior behavior) until measured.
    mount_lag_s: float = 0.0

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


class TrackingRunaway(RuntimeError):
    """Raised by run_tracking_loop when pointing error exceeds
    TrackingConfig.runaway_stop_deg -- the mount is diverging, not
    following. run_tracking_loop stops the mount before raising."""


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
    while (stop_event is None or not stop_event.is_set()) and time.monotonic() - t_loop_start < duration_s:
        now_wall = time.time()
        delta_t_s, perp_sign = offsets.snapshot()
        t_query = now_wall + delta_t_s + cfg.mount_lag_s

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

        ra_dir = _pick_direction(ra_rate_x_clamped, axis_signs.ra, "e", "w")
        dec_dir = _pick_direction(dec_rate_x_clamped, axis_signs.dec, "n", "s")
        mount.set_rate(abs(ra_rate_x_clamped))
        mount.move(ra_dir)
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
            _check_limits(mount, axis_signs)

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


def _check_limits(mount: Mount, axis_signs: AxisSigns) -> None:
    try:
        raw = mount.get_tracking_status()
    except ProtocolError as exc:
        print(f"[warn] bad :GAT# reply during status check: {exc}", file=sys.stderr)
        return
    code = parse_error(raw)
    if code in LIMIT_ERROR_CODES:
        print(f"[SAFETY] :GAT# reports error {code} (5=below horizon, 6=below altitude limit, "
              f"8=meridian crossed) — tracking may have silently stopped on the mount side", file=sys.stderr)

    # Deliberately NOT calling axis_signs.update_pier_side() here anymore
    # -- tried and reverted, see AxisSigns' docstring for the full
    # account. A real incident: this fired during live ISS tracking and
    # the resulting DEC sign flip caused a genuine ~35 deg divergence
    # (correctly caught by the runaway guard below, but the flip is what
    # caused it, not a real calibration or mount problem). :Gm#'s
    # relationship to true mechanical pier state during continuous
    # tracking (as opposed to a discrete :MS# GOTO, confirmed correct)
    # is unresolved.


TRACKING_CSV_FIELDS = [
    "t_mono", "t_utc", "target_ra_deg", "target_dec_deg", "actual_ra_deg", "actual_dec_deg",
    "ra_rate_x", "dec_rate_x", "delta_t_s", "perp_pulse",
]
