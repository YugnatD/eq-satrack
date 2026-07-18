#!/usr/bin/env python3
"""Track the next visible ISS pass with a feedforward rate-control loop.

Precomputes the whole pass's RA/DEC(t) trajectory from a fresh TLE before it
starts, then pushes per-axis rates to the mount at --loop-hz, adjustable
live via the keyboard (see am5/live_input.py: [ ] { } for timing, a/d for a
perpendicular nudge, q to stop early). Everything this depends on
(per-axis :Rv latching, additive tracking, smooth relatch on rate and
direction changes) was empirically validated against real hardware by
characterize.py — see that script's docstring for the findings.

This script does NOT slew the mount to the pass's starting position: point
it there by hand (the position is printed before the confirmation prompt),
same as picking the starting pier side that avoids a meridian flip mid-pass
on this German equatorial mount (also printed).

Usage:
    python3 track_pass.py --mock --start-immediately    # dry run against the simulator, right now
    python3 track_pass.py --serial /dev/ttyACM0          # real hardware, serial
    python3 track_pass.py --tcp 192.168.4.1:4030          # real hardware, WiFi
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from skyfield.api import wgs84

from am5.cli import add_connection_args, build_transport
from am5.ephemeris import compute_trajectory, find_next_pass, load_iss_tle, meridian_crossings
from am5.live_input import KeyboardInput
from am5.logging_utils import open_csv
from am5.mount import Mount
from am5.safety import SafetyGuard, confirm_ready_to_track
from am5.tracker import TRACKING_CSV_FIELDS, LiveOffsets, TrackingConfig, calibrate_directions, run_tracking_loop


def _wait_until(target: datetime, label: str) -> None:
    while True:
        remaining = (target - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            print(file=sys.stderr)
            return
        print(f"\r  waiting for {label}: T-{remaining:6.0f}s   ", end="", file=sys.stderr)
        time.sleep(min(remaining, 5.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_connection_args(parser)
    parser.add_argument("--site-lat", type=float, default=46.18, help="observer latitude, degrees")
    parser.add_argument("--site-lon", type=float, default=6.14, help="observer longitude, degrees")
    parser.add_argument("--site-elevation-m", type=float, default=0.0, help="observer elevation above sea level, meters")
    parser.add_argument("--horizon-deg", type=float, default=10.0, help="minimum elevation to count as a pass")
    parser.add_argument("--lookahead-hours", type=float, default=24.0, help="how far ahead to search for the next pass")
    parser.add_argument("--lead-s", type=float, default=10.0, help="start tracking this many seconds before AOS")
    parser.add_argument("--tle-cache", default="logs/iss.tle", help="local TLE cache path")
    parser.add_argument("--tle-max-age-hours", type=float, default=24.0, help="refetch the TLE if the cache is older than this")
    parser.add_argument("--loop-hz", type=float, default=20.0, help="tracking loop rate")
    parser.add_argument("--watchdog-timeout", type=float, default=5.0, help="seconds of silence before auto :Q#")
    parser.add_argument("--out-dir", default="logs", help="directory for CSV telemetry")
    parser.add_argument("--skip-confirm", action="store_true",
                         help="skip the READY TO TRACK confirmation (mock runs skip it automatically)")
    parser.add_argument("--start-immediately", action="store_true",
                         help="shift the whole precomputed pass to start right now instead of waiting for "
                              "the real AOS — for dry runs, or jumping into a synthetic pass for testing")
    parser.add_argument("--duration-cap-s", type=float, default=None,
                         help="stop after at most this many seconds, even if the pass isn't over (testing convenience)")
    args = parser.parse_args()

    site = wgs84.latlon(args.site_lat, args.site_lon, elevation_m=args.site_elevation_m)

    print("fetching TLE...", file=sys.stderr)
    satellite = load_iss_tle(Path(args.tle_cache), max_age_hours=args.tle_max_age_hours)
    print(f"  {satellite.name}, epoch {satellite.epoch.utc_iso()}", file=sys.stderr)

    window = find_next_pass(satellite, site, horizon_deg=args.horizon_deg, lookahead_hours=args.lookahead_hours)
    t_start = window.t_rise - timedelta(seconds=args.lead_s)

    # compute_trajectory must run against the REAL, unshifted pass window --
    # it queries the satellite's actual SGP4 position at whatever calendar
    # times it's given, so shifting t_start/window to "now" BEFORE this call
    # (the old order) computed the ISS's real position at the wrong, shifted
    # times instead of the real pass's actual geometry. --start-immediately
    # only relabels WHEN the (correctly-computed) trajectory happens on the
    # wall clock -- it must never change WHERE the satellite actually is at
    # each sample.
    trajectory = compute_trajectory(satellite, site, t_start, window.t_set, step_s=1.0 / args.loop_hz)

    if args.start_immediately:
        # Regression fix: this used to shift t_start/window BEFORE the
        # compute_trajectory call above (wrong geometry, see its own new
        # comment) AND ALSO shift trajectory.t_unix afterward -- a double
        # application that left the trajectory's active window hours away
        # from real "now" (confirmed: t_unix[0] landed 2 hours in the past
        # for a pass 2 hours out). run_tracking_loop queries the trajectory
        # at real wall-clock time, so with the window never overlapping
        # "now" at all, Trajectory.interpolate's own out-of-window handling
        # (rates zeroed, position clamped to the boundary -- see its own
        # docstring) meant the mount just sat motionless for the whole run,
        # every single time this flag was used -- exactly the script's own
        # top-of-file usage example for dry-running against the mock.
        # Shift applied exactly ONCE now, uniformly, to t_start/window (for
        # display) and trajectory.t_unix (for run_tracking_loop's own
        # wall-clock queries) together, after the real geometry above is
        # already locked in.
        shift = datetime.now(timezone.utc) - t_start
        t_start += shift
        window = dataclasses.replace(
            window,
            t_rise=window.t_rise + shift,
            t_culminate=window.t_culminate + shift,
            t_set=window.t_set + shift,
        )
        trajectory.t_unix = trajectory.t_unix + shift.total_seconds()

    duration_s = (window.t_set - window.t_rise).total_seconds()
    print(f"pass: rise {window.t_rise.isoformat()}  culminate {window.t_culminate.isoformat()} "
          f"(max el {window.max_elevation_deg:.1f} deg)  set {window.t_set.isoformat()}  "
          f"({duration_s:.0f}s)", file=sys.stderr)

    crossings = meridian_crossings(trajectory)
    if crossings:
        print(f"  MERIDIAN CROSSING during this pass at {', '.join(t.isoformat() for t in crossings)} — "
              f"start on the pier side that avoids a flip mid-pass", file=sys.stderr)
    else:
        print("  no meridian crossing during this pass", file=sys.stderr)

    start_ra, start_dec, _, _ = trajectory.interpolate(t_start.timestamp())
    print(f"  point the mount at RA={(start_ra % 360.0) / 15.0:.4f}h DEC={start_dec:+.4f} deg before continuing",
          file=sys.stderr)

    if not args.mock and not args.skip_confirm:
        confirm_ready_to_track()

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"logging to {out_dir}", file=sys.stderr)

    transport = build_transport(args)
    mount = Mount(transport)
    safety = SafetyGuard(mount, watchdog_timeout=args.watchdog_timeout)

    fh = None
    try:
        print(f"firmware: {mount.get_version()}", file=sys.stderr)
        axis_signs = calibrate_directions(mount, safety=safety)
        print(f"  calibrated axis signs: ra={axis_signs.ra:+.0f} dec={axis_signs.dec:+.0f}", file=sys.stderr)

        _wait_until(t_start, "AOS")

        remaining_s = max(0.0, (window.t_set - datetime.now(timezone.utc)).total_seconds())
        if args.duration_cap_s is not None:
            remaining_s = min(remaining_s, args.duration_cap_s)
        if remaining_s <= 0:
            print("pass window already elapsed, nothing to track", file=sys.stderr)
            return

        csv_writer, fh = open_csv(out_dir / "tracking.csv", TRACKING_CSV_FIELDS)
        offsets = LiveOffsets()
        stop_event = threading.Event()
        config = TrackingConfig(loop_hz=args.loop_hz)

        with KeyboardInput(offsets, on_quit=stop_event.set):
            run_tracking_loop(mount, safety, trajectory, axis_signs, offsets, csv_writer,
                               duration_s=remaining_s, config=config, stop_event=stop_event)
    finally:
        mount.stop()
        safety.notify_command(movement_active=False)
        safety.shutdown()
        if fh is not None:
            fh.close()
        mount.close()

    print("\npass complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
