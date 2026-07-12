#!/usr/bin/env python3
"""AM5 characterization rig.

Answers, against real hardware (or the mock, for development):

  (a) Is :Rv a global register or latched per-axis at :Me/:Mw/:Mn/:Ms time?
  (b) What's the step response (ramp time constant) to a :Rv change mid-slew?
  (c) What's the round-trip latency of :GMEQ#, and the delay before :Me#
      produces detectable motion?
  (d) Does :Me# add to sidereal tracking or replace it?
  (e) Does :Rv1440.00# actually deliver ~6.0 deg/s?
  (f) Given :Rv is per-axis-latched (per (a)/(b)), does re-issuing :Me# with a
      new :Rv while the axis is already moving relatch smoothly, or does it
      stop and restart the axis? Decides whether a 20Hz feedforward tracking
      loop can just keep calling :Rv+:Me every tick without jerking.

Usage:
    python3 characterize.py --mock                      # develop against the simulator
    python3 characterize.py --serial /dev/ttyACM0        # real hardware, serial
    python3 characterize.py --tcp 192.168.4.1:4030       # real hardware, WiFi
    python3 characterize.py --mock --tests a,d            # run a subset
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from am5.angles import circular_diff_deg, circular_diff_hours, unwrap_deg
from am5.cli import add_connection_args, build_transport
from am5.constants import SIDEREAL_DEG_PER_S
from am5.logging_utils import open_csv, utc_now_iso
from am5.mock_mount import MockConfig
from am5.mount import Mount
from am5.protocol import ProtocolError
from am5.safety import SafetyGuard, confirm_tube_removed



# --------------------------------------------------------------------------
# Polling / regression helpers
# --------------------------------------------------------------------------


@dataclass
class Sample:
    t_mono: float
    t_utc: str
    ra_deg: float
    dec_deg: float


def poll_radec(mount: Mount, duration_s: float, hz: float, csv_writer, tag: str = "") -> list[Sample]:
    """Poll :GMEQ# for `duration_s` seconds at up to `hz`, logging every
    successful sample to `csv_writer` and returning them."""
    period = 1.0 / hz
    samples: list[Sample] = []
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        t_req = time.monotonic()
        try:
            radec = mount.get_radec()
            s = Sample(t_req, utc_now_iso(), radec.ra_hours * 15.0, radec.dec_deg)
            samples.append(s)
            csv_writer.writerow(
                {"t_mono": s.t_mono, "t_utc": s.t_utc, "ra_deg": s.ra_deg, "dec_deg": s.dec_deg, "tag": tag}
            )
        except ProtocolError as exc:
            print(f"[warn] bad :GMEQ# reply during poll: {exc}", file=sys.stderr)
        elapsed = time.monotonic() - t_req
        time.sleep(max(0.0, period - elapsed))
    return samples


# The mount cannot physically exceed 1440x sidereal (~6.0 deg/s); a step
# implying more than this is a corrupted reply that still happened to parse
# (e.g. a stale byte spliced onto a fresh one), not real motion. A single
# such sample can otherwise swing an entire least-squares fit (or make
# np.unwrap latch a spurious +-360 deg offset for every later sample).
MAX_PHYSICAL_RATE_DEG_S = 8.0


def drop_impossible_jumps(samples: list[Sample]) -> list[Sample]:
    cleaned: list[Sample] = []
    last: Sample | None = None
    dropped = 0
    for s in samples:
        if last is not None:
            dt = s.t_mono - last.t_mono
            d_ra = circular_diff_deg(s.ra_deg, last.ra_deg)
            d_dec = s.dec_deg - last.dec_deg
            if dt > 0 and (abs(d_ra) / dt > MAX_PHYSICAL_RATE_DEG_S or abs(d_dec) / dt > MAX_PHYSICAL_RATE_DEG_S):
                dropped += 1
                continue
        cleaned.append(s)
        last = s
    if dropped:
        print(f"  [warn] dropped {dropped}/{len(samples)} samples as physically impossible jumps", file=sys.stderr)
    return cleaned


def linear_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope dy/dt."""
    if len(t) < 2:
        return float("nan")
    a = np.vstack([t - t[0], np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(a, y, rcond=None)[0]
    return float(slope)


def windowed_velocity(t: np.ndarray, y: np.ndarray, half_window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference velocity over a +-half_window sample span, to see
    through single-sample quantization noise (e.g. AM3's 15" RA ticks) without
    smearing out a genuine stop/restart glitch as badly as a wide moving
    average would. Returns (t_mid, v) shorter than the input by 2*half_window."""
    n = len(t)
    if n <= 2 * half_window:
        return np.array([]), np.array([])
    t_mid = t[half_window:n - half_window]
    v = (y[2 * half_window:] - y[: n - 2 * half_window]) / (t[2 * half_window:] - t[: n - 2 * half_window])
    return t_mid, v


def deg_per_s_to_x_sidereal(rate_deg_s: float) -> float:
    return rate_deg_s / SIDEREAL_DEG_PER_S


def text_histogram(values: list[float], bins: int = 12, width: int = 40) -> str:
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    if lo == hi:
        hi = lo + 1e-9
    counts = [0] * bins
    step = (hi - lo) / bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / step))
        counts[idx] += 1
    max_count = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bucket_lo = lo + i * step
        bar = "#" * int(width * c / max_count)
        lines.append(f"  {bucket_lo * 1000:7.1f} ms | {bar} ({c})")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_a_rv_axis_mode(mount: Mount, safety: SafetyGuard, out_dir: Path) -> None:
    print("\n=== (a) :Rv global vs per-axis ===")
    csv_writer, fh = open_csv(out_dir / "a_rv_axis_mode.csv", ["t_mono", "t_utc", "ra_deg", "dec_deg", "tag"])
    try:
        mount.stop()
        safety.notify_command(movement_active=False)
        mount.set_tracking(False)
        time.sleep(0.2)

        mount.set_rate(239.0)
        mount.move("e")
        safety.notify_command(movement_active=True)
        time.sleep(0.5)  # let RA settle onto 239x before touching DEC

        mount.set_rate(90.0)
        mount.move("n")
        safety.notify_command(movement_active=True)

        samples = poll_radec(mount, duration_s=2.0, hz=15.0, csv_writer=csv_writer, tag="post_change")
    finally:
        mount.stop("e")
        mount.stop("n")
        safety.notify_command(movement_active=False)
        fh.close()

    samples = drop_impossible_jumps(samples)
    t = np.array([s.t_mono for s in samples])
    ra = unwrap_deg(np.array([s.ra_deg for s in samples]))
    dec = np.array([s.dec_deg for s in samples])
    ra_rate_x = deg_per_s_to_x_sidereal(linear_slope(t, ra))
    dec_rate_x = deg_per_s_to_x_sidereal(linear_slope(t, dec))

    print(f"  RA rate after DEC :Rv90 command:  {ra_rate_x:8.1f}x sidereal (commanded 239x)")
    print(f"  DEC rate:                         {dec_rate_x:8.1f}x sidereal (commanded 90x)")
    if abs(ra_rate_x - 239.0) < abs(ra_rate_x - 90.0):
        print("  => :Rv appears LATCHED PER-AXIS (RA kept its own rate).")
    else:
        print("  => :Rv appears GLOBAL (RA rate followed the DEC :Rv change).")


def test_b_step_response(mount: Mount, safety: SafetyGuard, out_dir: Path) -> None:
    print("\n=== (b) step response to :Rv change mid-slew ===")
    csv_writer, fh = open_csv(out_dir / "b_step_response.csv", ["t_mono", "t_utc", "ra_deg", "dec_deg", "tag"])
    try:
        mount.stop()
        safety.notify_command(movement_active=False)
        mount.set_tracking(False)
        time.sleep(0.2)

        mount.set_rate(50.0)
        mount.move("e")
        safety.notify_command(movement_active=True)
        pre_samples = poll_radec(mount, duration_s=1.0, hz=30.0, csv_writer=csv_writer, tag="pre_step")

        t_step = time.monotonic()
        mount.set_rate(200.0)
        post_samples = poll_radec(mount, duration_s=2.0, hz=30.0, csv_writer=csv_writer, tag="post_step")
    finally:
        mount.stop("e")
        safety.notify_command(movement_active=False)
        fh.close()

    pre_samples = drop_impossible_jumps(pre_samples)
    post_samples = drop_impossible_jumps(post_samples)
    pre_t = np.array([s.t_mono for s in pre_samples])
    pre_ra = unwrap_deg(np.array([s.ra_deg for s in pre_samples]))
    v0 = linear_slope(pre_t, pre_ra)

    t = np.array([s.t_mono for s in post_samples])
    ra = unwrap_deg(np.array([s.ra_deg for s in post_samples]))
    if len(t) < 3:
        print("  not enough samples after the step to fit a response")
        return
    # instantaneous velocity via central differences
    v = np.gradient(ra, t)
    v1 = float(np.median(v[-5:])) if len(v) >= 5 else float(v[-1])

    target_v = v0 + 0.63 * (v1 - v0)
    tau_s = float("nan")
    for ti, vi in zip(t, v):
        if (v1 >= v0 and vi >= target_v) or (v1 < v0 and vi <= target_v):
            tau_s = ti - t_step
            break

    print(f"  v0 (50x): {deg_per_s_to_x_sidereal(v0):7.1f}x sidereal")
    print(f"  v1 (200x settled): {deg_per_s_to_x_sidereal(v1):7.1f}x sidereal")
    print(f"  time to 63% of step: {tau_s * 1000:.1f} ms" if tau_s == tau_s else "  time to 63%: not reached in window")


def test_c_latency(mount: Mount, safety: SafetyGuard, out_dir: Path, n: int = 200) -> None:
    print("\n=== (c) round-trip latency ===")
    csv_writer, fh = open_csv(out_dir / "c_latency.csv", ["t_mono", "t_utc", "latency_s", "kind"])
    latencies: list[float] = []
    try:
        mount.stop()
        safety.notify_command(movement_active=False)
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                mount.get_radec()
            except ProtocolError:
                pass
            dt = time.perf_counter() - t0
            latencies.append(dt)
            csv_writer.writerow({"t_mono": t0, "t_utc": utc_now_iso(), "latency_s": dt, "kind": "GMEQ"})

        # delay between :Me# and first detectable movement
        move_delays: list[float] = []
        for _ in range(10):
            mount.set_tracking(False)
            mount.stop("e")
            time.sleep(0.3)
            baseline = mount.get_radec().ra_hours
            mount.set_rate(100.0)
            t_cmd = time.perf_counter()
            mount.move("e")
            safety.notify_command(movement_active=True)
            t_detect = None
            t_giveup = time.perf_counter() + 1.0
            while time.perf_counter() < t_giveup:
                ra = mount.get_radec().ra_hours
                if abs(circular_diff_hours(ra, baseline)) * 15.0 * 3600 > 2.0:  # > 2 arcsec moved
                    t_detect = time.perf_counter()
                    break
            mount.stop("e")
            safety.notify_command(movement_active=False)
            if t_detect is not None:
                delay = t_detect - t_cmd
                move_delays.append(delay)
                csv_writer.writerow({"t_mono": t_cmd, "t_utc": utc_now_iso(), "latency_s": delay, "kind": "Me_first_motion"})
    finally:
        mount.stop()
        safety.notify_command(movement_active=False)
        fh.close()

    latencies_ms = [x * 1000 for x in latencies]
    print(f"  :GMEQ# round trip over {len(latencies)} samples:")
    print(f"    median {statistics.median(latencies_ms):.1f} ms   "
          f"p99 {np.percentile(latencies_ms, 99):.1f} ms   "
          f"mean {statistics.mean(latencies_ms):.1f} ms")
    print(text_histogram(latencies))
    if move_delays:
        print(f"  :Me# -> first detectable motion: median {statistics.median(move_delays) * 1000:.1f} ms "
              f"over {len(move_delays)} trials")


def test_d_tracking_addition(mount: Mount, safety: SafetyGuard, out_dir: Path) -> None:
    print("\n=== (d) does :Me# add to sidereal or replace it? ===")
    csv_writer, fh = open_csv(out_dir / "d_tracking_addition.csv", ["t_mono", "t_utc", "ra_deg", "dec_deg", "tag"])

    def measure_rate(tracking_on: bool) -> float:
        mount.stop()
        safety.notify_command(movement_active=False)
        mount.set_tracking(tracking_on)
        time.sleep(0.3)
        mount.set_rate(60.0)
        mount.move("e")
        safety.notify_command(movement_active=True)
        time.sleep(0.5)  # let the ramp settle so the transient doesn't bias the slope fit
        samples = poll_radec(mount, duration_s=2.5, hz=20.0, csv_writer=csv_writer,
                              tag=f"tracking_{'on' if tracking_on else 'off'}")
        mount.stop("e")
        safety.notify_command(movement_active=False)
        samples = drop_impossible_jumps(samples)
        t = np.array([s.t_mono for s in samples])
        ra = unwrap_deg(np.array([s.ra_deg for s in samples]))
        return linear_slope(t, ra)

    try:
        rate_on = measure_rate(True)
        rate_off = measure_rate(False)
    finally:
        mount.stop()
        safety.notify_command(movement_active=False)
        fh.close()

    diff = rate_on - rate_off
    print(f"  dRA/dt with tracking ON:  {deg_per_s_to_x_sidereal(rate_on):7.2f}x sidereal")
    print(f"  dRA/dt with tracking OFF: {deg_per_s_to_x_sidereal(rate_off):7.2f}x sidereal")
    print(f"  difference: {diff:.6f} deg/s ({deg_per_s_to_x_sidereal(diff):.2f}x sidereal, "
          f"1x sidereal = {SIDEREAL_DEG_PER_S:.6f} deg/s)")
    if abs(diff) < 0.3 * SIDEREAL_DEG_PER_S:
        print("  => :Me# ADDS on top of sidereal compensation (tracking state doesn't change net rate).")
    else:
        print("  => :Me# REPLACES sidereal compensation (tracking state shifts the net rate by ~1x sidereal).")


def test_e_max_speed(mount: Mount, safety: SafetyGuard, out_dir: Path) -> None:
    print("\n=== (e) max speed at :Rv1440.00# ===")
    csv_writer, fh = open_csv(out_dir / "e_max_speed.csv", ["t_mono", "t_utc", "ra_deg", "dec_deg", "tag"])
    try:
        mount.stop()
        safety.notify_command(movement_active=False)
        mount.set_tracking(False)
        time.sleep(0.2)

        mount.set_rate(1440.0)
        mount.move("e")
        safety.notify_command(movement_active=True)
        settle_samples = poll_radec(mount, duration_s=2.5, hz=30.0, csv_writer=csv_writer, tag="settling")
        samples = poll_radec(mount, duration_s=1.5, hz=30.0, csv_writer=csv_writer, tag="max_speed")
    finally:
        mount.stop("e")
        safety.notify_command(movement_active=False)
        fh.close()

    samples = drop_impossible_jumps(samples)
    t = np.array([s.t_mono for s in samples])
    ra = unwrap_deg(np.array([s.ra_deg for s in samples]))
    rate_deg_s = linear_slope(t, ra)
    nominal_deg_s = 1440.0 * SIDEREAL_DEG_PER_S
    print(f"  measured: {rate_deg_s:.3f} deg/s   nominal 1440x: {nominal_deg_s:.3f} deg/s   "
          f"ratio: {rate_deg_s / nominal_deg_s:.3f}")

    mid = len(t) // 2
    if mid >= 2:
        first_half = linear_slope(t[:mid], ra[:mid])
        second_half = linear_slope(t[mid:], ra[mid:])
        if second_half > first_half * 1.05:
            print(f"  [warn] still accelerating during the window ({first_half:.2f} -> {second_half:.2f} deg/s) "
                  f"— settle time is too short, this measurement understates true max speed")


def test_f_relatch_smoothness(mount: Mount, safety: SafetyGuard, out_dir: Path) -> None:
    print("\n=== (f) does re-issuing :Me# mid-slew relatch smoothly? ===")
    csv_writer, fh = open_csv(out_dir / "f_relatch_smoothness.csv", ["t_mono", "t_utc", "ra_deg", "dec_deg", "tag"])
    try:
        mount.stop()
        safety.notify_command(movement_active=False)
        mount.set_tracking(False)
        time.sleep(0.2)

        mount.set_rate(100.0)
        mount.move("e")
        safety.notify_command(movement_active=True)
        pre_samples = poll_radec(mount, duration_s=0.6, hz=60.0, csv_writer=csv_writer, tag="pre_relatch")

        t_relatch = time.monotonic()
        mount.set_rate(300.0)
        mount.move("e")  # same direction, already moving — does this relatch cleanly?
        safety.notify_command(movement_active=True)
        post_samples = poll_radec(mount, duration_s=1.2, hz=60.0, csv_writer=csv_writer, tag="post_relatch")
    finally:
        mount.stop("e")
        safety.notify_command(movement_active=False)
        fh.close()

    samples = drop_impossible_jumps(pre_samples + post_samples)
    t = np.array([s.t_mono for s in samples])
    ra = unwrap_deg(np.array([s.ra_deg for s in samples]))
    if len(t) < 12:
        print("  not enough samples to analyze")
        return

    t_mid, v = windowed_velocity(t, ra, half_window=2)
    if len(t_mid) < 6:
        print("  not enough samples to analyze")
        return

    v_before = float(np.median(v[t_mid < t_relatch][-5:])) if np.any(t_mid < t_relatch) else float("nan")
    tail = v[t_mid > t_relatch + 0.8]
    v_after = float(np.median(tail)) if len(tail) else float(v[-1])
    near = (t_mid >= t_relatch - 0.05) & (t_mid <= t_relatch + 0.5)
    v_min_near = float(np.min(v[near])) if np.any(near) else float("nan")

    print(f"  v_before (100x): {deg_per_s_to_x_sidereal(v_before):7.1f}x sidereal")
    print(f"  v_min in [relatch, relatch+0.5s]: {deg_per_s_to_x_sidereal(v_min_near):7.1f}x sidereal")
    print(f"  v_after (300x settled): {deg_per_s_to_x_sidereal(v_after):7.1f}x sidereal")
    print("  velocity trace around the reissue (x sidereal):")
    for tm, vv in zip(t_mid, v):
        if t_relatch - 0.1 <= tm <= t_relatch + 0.6:
            marker = " <-- :Rv300#+:Me# sent" if tm >= t_relatch and (tm - t_relatch) < (t_mid[1] - t_mid[0]) else ""
            print(f"    t={tm - t_relatch:+.3f}s  v={deg_per_s_to_x_sidereal(vv):7.1f}x{marker}")

    if v_min_near < 0.4 * v_before:
        print("  => STOP/RESTART: velocity dips toward zero at the reissue before climbing back up.")
        print("     A 20Hz loop naively re-sending :Rv+:Me every tick will jerk the axis.")
    elif v_min_near >= 0.8 * v_before:
        print("  => SMOOTH: velocity climbs monotonically from 100x to 300x, no dip.")
        print("     Re-issuing :Rv+:Me every tick should be safe for continuous feedforward tracking.")
    else:
        print("  => AMBIGUOUS: partial dip — inspect f_relatch_smoothness.csv before relying on this.")


# Constant acceleration observed while ramping RA up to 1440x in test (e):
# ~6.0 deg/s reached in ~1.6s from rest. Used only as a reference to judge
# whether a direction reversal looks like one continuous ramp through zero
# or a stop-then-restart with a dead zone in between.
MEASURED_ACCEL_DEG_S2 = 3.8


def test_g_direction_reversal(mount: Mount, safety: SafetyGuard, out_dir: Path) -> None:
    print("\n=== (g) does reversing direction (E->W) mid-slew ramp through zero, or stop first? ===")
    csv_writer, fh = open_csv(out_dir / "g_direction_reversal.csv", ["t_mono", "t_utc", "ra_deg", "dec_deg", "tag"])
    try:
        mount.stop()
        safety.notify_command(movement_active=False)
        mount.set_tracking(False)
        time.sleep(0.2)

        mount.set_rate(100.0)
        mount.move("e")
        safety.notify_command(movement_active=True)
        pre_samples = poll_radec(mount, duration_s=0.6, hz=60.0, csv_writer=csv_writer, tag="pre_reversal")

        t_reverse = time.monotonic()
        mount.set_rate(150.0)
        mount.move("w")  # opposite direction, already moving east — smooth ramp through zero, or stop first?
        safety.notify_command(movement_active=True)
        post_samples = poll_radec(mount, duration_s=1.2, hz=60.0, csv_writer=csv_writer, tag="post_reversal")
    finally:
        mount.stop("e")
        mount.stop("w")
        safety.notify_command(movement_active=False)
        fh.close()

    samples = drop_impossible_jumps(pre_samples + post_samples)
    t = np.array([s.t_mono for s in samples])
    ra = unwrap_deg(np.array([s.ra_deg for s in samples]))
    if len(t) < 12:
        print("  not enough samples to analyze")
        return

    t_mid, v = windowed_velocity(t, ra, half_window=2)
    if len(t_mid) < 6:
        print("  not enough samples to analyze")
        return

    v_before = float(np.median(v[t_mid < t_reverse][-5:])) if np.any(t_mid < t_reverse) else float("nan")
    tail = v[t_mid > t_reverse + 0.8]
    v_after = float(np.median(tail)) if len(tail) else float(v[-1])

    print(f"  v_before (100x east): {deg_per_s_to_x_sidereal(v_before):7.1f}x sidereal")
    print(f"  v_after (150x west settled): {deg_per_s_to_x_sidereal(v_after):7.1f}x sidereal")
    print("  velocity trace around the reversal (x sidereal):")
    for tm, vv in zip(t_mid, v):
        if t_reverse - 0.1 <= tm <= t_reverse + 0.6:
            marker = " <-- :Rv150#+:Mw# sent" if tm >= t_reverse and (tm - t_reverse) < (t_mid[1] - t_mid[0]) else ""
            print(f"    t={tm - t_reverse:+.3f}s  v={deg_per_s_to_x_sidereal(vv):7.1f}x{marker}")

    near_zero_band = 0.15 * max(abs(v_before), abs(v_after))
    near = (t_mid >= t_reverse - 0.05) & (t_mid <= t_reverse + 0.8) & (np.abs(v) < near_zero_band)
    near_zero_duration = float(np.ptp(t_mid[near])) if np.count_nonzero(near) >= 2 else 0.0
    expected_transition_s = (abs(v_before) + abs(v_after)) / MEASURED_ACCEL_DEG_S2

    print(f"  time spent near zero velocity: {near_zero_duration * 1000:.0f} ms "
          f"(single continuous ramp at ~{MEASURED_ACCEL_DEG_S2} deg/s^2 would take ~{expected_transition_s * 1000:.0f} ms total)")
    if near_zero_duration > 3 * expected_transition_s + 0.1:
        print("  => STOP-THEN-RESTART: a dead zone near zero velocity, well beyond what a continuous ramp explains.")
        print("     tracker.py must send :Q<dir># before reversing direction on an axis.")
    else:
        print("  => CONTINUOUS RAMP THROUGH ZERO: no meaningful dead zone.")
        print("     tracker.py can re-issue :Rv+:M<newdir># directly on a direction change, no :Q<dir># needed.")


# Consecutive-sample movement below this is treated as "arrived" for the
# GOTO-completion heuristic in test (h) — no encoders, so this is the only
# signal available (mirrors what a GUI GOTO button would have to do).
GOTO_ARRIVED_THRESHOLD_ARCSEC = 5.0


MS_REPLY_MEANING = {
    0: "slewing",
    1: "target below horizon",
    2: "target below the altitude limit",
    -7: "e7: time/location not synced — mount needs :SMTI#/:SMGE# or St/Sg/SC/SL/SG before any GOTO",
}


def describe_ms_result(result: int) -> str:
    if result in MS_REPLY_MEANING:
        return MS_REPLY_MEANING[result]
    if result < 0:
        return f"e{-result}: undocumented error code"
    return "undocumented reply code"


def test_h_goto_characterization(mount: Mount, safety: SafetyGuard, out_dir: Path) -> None:
    print("\n=== (h) :Sr#/:Sd#/:MS# GOTO characterization ===")
    print("  NOTE: deliberately not testing a GOTO that crosses the meridian in this test.")
    csv_writer, fh = open_csv(out_dir / "h_goto.csv", ["t_mono", "t_utc", "ra_deg", "dec_deg", "tag"])
    samples = []
    ms_result = None
    try:
        mount.stop()
        safety.notify_command(movement_active=False)
        time.sleep(0.5)  # let any residual motion from a prior test's :Q# fully settle first
        baseline = mount.get_radec()
        target_ra_hours = baseline.ra_hours + 2.0 / 15.0  # +2 deg in RA
        target_dec_deg = baseline.dec_deg + (2.0 if baseline.dec_deg < 60.0 else -2.0)
        print(f"  baseline: RA={baseline.ra_hours:.4f}h DEC={baseline.dec_deg:+.4f} deg")
        print(f"  target:   RA={target_ra_hours:.4f}h DEC={target_dec_deg:+.4f} deg (+2 deg on each axis)")

        t_cmd = time.monotonic()
        ms_result = mount.goto(target_ra_hours, target_dec_deg)
        t_reply = time.monotonic()
        print(f"  :MS# result: {ms_result} ({describe_ms_result(ms_result)}) "
              f"after {(t_reply - t_cmd) * 1000:.1f} ms")
        if ms_result != 0:
            print("  mount did not accept the slew — skipping arrival polling for this target.")
        else:
            safety.notify_command(movement_active=True)
            samples = poll_radec(mount, duration_s=5.0, hz=30.0, csv_writer=csv_writer, tag="goto")
    finally:
        mount.stop()
        safety.notify_command(movement_active=False)

    samples = drop_impossible_jumps(samples)
    if ms_result != 0:
        pass
    elif len(samples) < 5:
        print("  not enough samples to analyze arrival")
    else:
        arrived_idx = None
        for i in range(2, len(samples)):
            d1 = math.hypot(circular_diff_deg(samples[i].ra_deg, samples[i - 1].ra_deg),
                             samples[i].dec_deg - samples[i - 1].dec_deg) * 3600
            d2 = math.hypot(circular_diff_deg(samples[i - 1].ra_deg, samples[i - 2].ra_deg),
                             samples[i - 1].dec_deg - samples[i - 2].dec_deg) * 3600
            if d1 < GOTO_ARRIVED_THRESHOLD_ARCSEC and d2 < GOTO_ARRIVED_THRESHOLD_ARCSEC:
                arrived_idx = i
                break
        if arrived_idx is not None:
            t_arrive = samples[arrived_idx].t_mono - t_cmd
            final = mount.get_radec()
            err_ra = abs(circular_diff_hours(final.ra_hours, target_ra_hours)) * 15 * 3600
            err_dec = abs(final.dec_deg - target_dec_deg) * 3600
            print(f"  arrival detected at t+{t_arrive:.2f}s "
                  f"(heuristic: <{GOTO_ARRIVED_THRESHOLD_ARCSEC}\"/sample x2), "
                  f"final error: RA {err_ra:.1f}\" DEC {err_dec:.1f}\"")
        else:
            print(f"  arrival NOT detected within the {5.0}s window — either still moving, or the "
                  f"heuristic is too strict for this mount. Inspect h_goto.csv before trusting it in a GUI.")

    print("  sub-test: does :Q# stop a GOTO cleanly mid-flight?")
    back_result = mount.goto(baseline.ra_hours, baseline.dec_deg)  # head back toward the start
    if back_result != 0:
        print(f"  :MS# rejected the return trip too ({back_result}, "
              f"{describe_ms_result(back_result)}) — skipping the :Q# sub-test.")
        fh.close()
        return
    safety.notify_command(movement_active=True)
    time.sleep(0.15)
    mount.stop()
    safety.notify_command(movement_active=False)
    time.sleep(0.3)
    after_stop = [mount.get_radec()]
    for _ in range(4):
        time.sleep(0.1)
        after_stop.append(mount.get_radec())
    moved_arcsec = max(
        math.hypot(circular_diff_hours(r.ra_hours, after_stop[0].ra_hours) * 15, r.dec_deg - after_stop[0].dec_deg) * 3600
        for r in after_stop[1:]
    )
    print(f"  after :Q#, position drifted <= {moved_arcsec:.1f}\" over the next 0.5s "
          f"(should be ~0 if :Q# actually stopped the GOTO)")

    fh.close()


TESTS = {
    "a": test_a_rv_axis_mode,
    "b": test_b_step_response,
    "c": test_c_latency,
    "d": test_d_tracking_addition,
    "e": test_e_max_speed,
    "f": test_f_relatch_smoothness,
    "g": test_g_direction_reversal,
    "h": test_h_goto_characterization,
}


# --------------------------------------------------------------------------
# CLI / wiring
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_connection_args(parser)
    parser.add_argument("--tests", default="a,b,c,d,e,f", help="comma-separated subset of a,b,c,d,e,f,g")
    parser.add_argument("--out-dir", default="logs", help="directory for CSV logs")
    parser.add_argument("--watchdog-timeout", type=float, default=5.0, help="seconds of silence before auto :Q#")
    parser.add_argument("--mock-rv-mode", choices=["global", "per_axis"], default="per_axis",
                         help="mock only: simulate which :Rv hypothesis")
    parser.add_argument("--mock-tracking-replaces", action="store_true",
                         help="mock only: simulate :Me# replacing sidereal instead of adding to it")
    parser.add_argument("--skip-confirm", action="store_true",
                         help="skip the tube-removed confirmation (mock runs skip it automatically)")
    args = parser.parse_args()

    if not args.mock and not args.skip_confirm:
        confirm_tube_removed()

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"logging to {out_dir}")

    mock_config = MockConfig(rv_mode=args.mock_rv_mode, tracking_adds=not args.mock_tracking_replaces)
    transport = build_transport(args, mock_config=mock_config)
    mount = Mount(transport)
    safety = SafetyGuard(mount, watchdog_timeout=args.watchdog_timeout)

    try:
        print(f"firmware: {mount.get_version()}")
        selected = [t.strip() for t in args.tests.split(",") if t.strip()]
        for key in selected:
            if key not in TESTS:
                print(f"unknown test {key!r}, skipping", file=sys.stderr)
                continue
            TESTS[key](mount, safety, out_dir)
    finally:
        mount.stop()
        safety.notify_command(movement_active=False)
        safety.shutdown()
        mount.close()

    print("\ndone.")


if __name__ == "__main__":
    main()
