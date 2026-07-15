"""Background thread that owns the Mount and serializes every command sent
to it. Nothing in am5/gui/panels.py or app.py should ever touch a Mount
directly — Transport objects are not safe for concurrent access (two
commands' write/read pairs interleaving on the wire corrupts both), which
this project has hit for real earlier in development.

The one deliberate exception is emergency_stop(): it bypasses the command
queue entirely and writes straight to the transport, because a stop must
never wait behind whatever else is queued. Mount.emergency_stop() is
already designed for exactly this (best-effort, never raises, safe to call
from any thread — originally built for signal handlers).
"""

from __future__ import annotations

import csv as csv_module
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from am5.angles import angular_separation_deg, circular_diff_hours
from am5.clock_sync import check_clock_sync
from am5.ephemeris import Trajectory
from am5.mock_mount import MockConfig, MockMount
from am5.mount import Mount
from am5.protocol import ProtocolError
from am5.safety import SafetyGuard
from am5.tracker import LIMIT_CODE_MEANINGS, TRACKING_CSV_FIELDS, AxisSigns, LiveOffsets, TrackingConfig, _pick_direction, calibrate_directions, measure_mount_lag, run_tracking_loop
from am5.transport import SerialTransport, TCPTransport, Transport

# Same heuristic as characterize.py test (f): no encoders, so "arrived" can
# only ever mean "stopped changing", never a hardware-confirmed event.
GOTO_ARRIVED_THRESHOLD_ARCSEC = 5.0
GOTO_POLL_TIMEOUT_S = 15.0
IDLE_POLL_INTERVAL_S = 0.4  # ~2Hz -- see _run's own comment for why this isn't a blocking sleep anymore

MS_REPLY_MEANING = {
    0: "slewing",
    1: "target below horizon",
    2: "target below the altitude limit",
    # e<code># forms, per the official protocol doc's :MS# error table
    # (docs/ZWO Mount Serial Communication Protocol_v1.7.pdf) -- codes 1/2
    # above were confirmed bare (no 'e' prefix) against real hardware
    # earlier in this project; these -N entries cover the 'e'-prefixed
    # forms so an unexpected one doesn't just show as a bare "e3" fallback.
    -1: "parameter out of range",
    -2: "parameter format error",
    -3: "mount busy (already homing/slewing/doing a GOTO)",
    -4: "equipment moving",
    -5: "target below horizon",
    -6: "target below the altitude limit",
    -7: "time/location not synced",
    -8: "meridian passed during tracking",
}
# How far the mount is allowed to land from the requested target before
# flagging it -- real backlash/settling error is arcseconds to a few
# arcmin, not degrees. Landing degrees away from a GOTO that reported
# success (not e3#/e7# etc.) points at a real mechanical problem (stall,
# obstruction, bad calibration), not a software targeting bug -- see the
# incident that prompted this: 3 identical GOTOs to the same target
# converged from 70+ deg of error down to a few arcsec over 3 clicks.
GOTO_MISMATCH_WARN_DEG = 1.0

# jog_goto: closed-loop proportional approach, gentle enough to not need a
# tube-removed style confirmation (same order of magnitude as manual jog
# rates the operator already drives unattended).
JOG_GOTO_KP_RATE_X_PER_DEG = 200.0
JOG_GOTO_MAX_RATE_X = 400.0
JOG_GOTO_ARRIVED_ARCSEC = 10.0
JOG_GOTO_TIMEOUT_S = 180.0
# Abort a manual GOTO if the pointing error grows this far past its best --
# it's jogging away from the target (wrong axis-sign calibration), not
# converging. Comfortably above proportional-controller overshoot.
JOG_GOTO_DIVERGENCE_DEG = 3.0


@dataclass
class WorkerEvent:
    kind: str
    payload: dict = field(default_factory=dict)


class MountWorker:
    """One dedicated thread, one Mount. Commands go in via the public
    methods below (thread-safe, callable from the Tk main thread); results
    and live telemetry come out via `events` (a queue.Queue), which the GUI
    polls with `root.after(...)` — never blocks the Tk mainloop."""

    def __init__(self) -> None:
        self.events: "queue.Queue[WorkerEvent]" = queue.Queue()
        self._commands: "queue.Queue[tuple[str, dict]]" = queue.Queue()
        self._mount: Mount | None = None
        # Set at connect time only for kind="mock" (see _handle_connect),
        # None otherwise -- lets inject_training_pointing_error refuse to
        # act on a real mount even if a caller ever gets this wrong, since
        # it checks this instead of trusting the GUI's own last-requested
        # connection kind.
        self._mock_mount: MockMount | None = None
        self._safety: SafetyGuard | None = None
        self._connected = threading.Event()
        self._idle_poll_enabled = threading.Event()
        self._parked = False
        self._tracking_stop_event: threading.Event | None = None
        # Set by emergency_stop() to break the closed-loop motion handlers
        # (jog_goto, calibrate) that would otherwise re-issue a move command
        # a fraction of a second after the emergency :Q#, silently
        # overriding it. Cleared at the start of each such handler.
        self._abort = threading.Event()
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- public, thread-safe API ------------------------------------------------

    def connect(
        self, kind: str, address: str = "", mock_seed: int | None = None,
        latitude_deg: float = 46.18, longitude_deg: float = 6.14,
    ) -> None:
        self._commands.put(("connect", {
            "kind": kind, "address": address, "mock_seed": mock_seed,
            "latitude_deg": latitude_deg, "longitude_deg": longitude_deg,
        }))

    def disconnect(self) -> None:
        self._commands.put(("disconnect", {}))

    def jog_start(self, direction: str, rate_x: float) -> None:
        self._commands.put(("jog_start", {"direction": direction, "rate_x": rate_x}))

    def jog_stop(self, direction: str) -> None:
        self._commands.put(("jog_stop", {"direction": direction}))

    def jog_goto(self, ra_hours: float, dec_deg: float, axis_signs: AxisSigns) -> None:
        """Closed-loop approach to a target using jog primitives (:Rv#+:Me#
        etc.), never :MS# -- so it never triggers the firmware's autonomous
        pier-side choice. Takes the shortest RA/DEC path, so it preserves
        whatever side the mount is already on (get it there by hand-jogging
        first if you need a specific side, then use this to finish
        precisely) -- see the incident this was built for: :MS# ignored the
        operator's manually-set starting side entirely."""
        self._commands.put(("jog_goto", {"ra_hours": ra_hours, "dec_deg": dec_deg, "axis_signs": axis_signs}))

    def stop_all(self) -> None:
        # Set the abort flag directly (not just via the queued handler): a
        # running jog_goto/calibrate blocks the command queue, so a queued
        # stop_all wouldn't be processed until that loop finished on its own.
        self._abort.set()
        self._commands.put(("stop_all", {}))

    def park(self) -> None:
        self._commands.put(("park", {}))

    def park_native(self) -> None:
        self._commands.put(("park_native", {}))

    def unpark(self) -> None:
        self._commands.put(("unpark", {}))

    def set_tracking(self, on: bool) -> None:
        self._commands.put(("set_tracking", {"on": on}))

    def set_altitude_limits(self, enabled: bool) -> None:
        self._commands.put(("set_altitude_limits", {"enabled": enabled}))

    def goto(self, ra_hours: float, dec_deg: float) -> None:
        self._commands.put(("goto", {"ra_hours": ra_hours, "dec_deg": dec_deg}))

    def sync(self, ra_hours: float, dec_deg: float) -> None:
        """:Sr#+:Sd#+:CM# — tells the mount its current position IS
        (ra_hours, dec_deg), no motion. For correcting the mount's belief
        about where it's pointed against a manually-centered, known
        reference (a star) -- not a substitute for GOTO, and syncing on
        anything not actually centered introduces a wrong offset instead
        of fixing one."""
        self._commands.put(("sync", {"ra_hours": ra_hours, "dec_deg": dec_deg}))

    def calibrate(self) -> None:
        self._commands.put(("calibrate", {}))

    def inject_training_pointing_error(self, ra_bias_deg: float, dec_bias_deg: float) -> None:
        """Mock-only training aid: see MockMount.inject_pointing_error's
        own docstring. Silently ignored (with a log line) if the
        currently connected mount isn't mock -- see _mock_mount's own
        comment for why this is checked on the worker thread against the
        actual connected transport, not trusted from the caller."""
        self._commands.put(("inject_training_pointing_error", {"ra_bias_deg": ra_bias_deg, "dec_bias_deg": dec_bias_deg}))

    def measure_mount_lag(self, rate_x: float = 100.0, duration_s: float = 1.5) -> None:
        self._commands.put(("measure_mount_lag", {"rate_x": rate_x, "duration_s": duration_s}))

    def start_tracking(
        self, trajectory: Trajectory, axis_signs: AxisSigns, offsets: LiveOffsets,
        csv_path: Path, duration_s: float, config: TrackingConfig | None = None,
    ) -> threading.Event:
        """Returns the stop_event for this session — call .set() on it (or
        MountWorker.stop_tracking()) to end the pass early. Thread-safe,
        does not go through the command queue."""
        stop_event = threading.Event()
        self._commands.put(("start_tracking", {
            "trajectory": trajectory, "axis_signs": axis_signs, "offsets": offsets,
            "csv_path": csv_path, "duration_s": duration_s, "config": config,
            "stop_event": stop_event,
        }))
        return stop_event

    def stop_tracking(self) -> None:
        if self._tracking_stop_event is not None:
            self._tracking_stop_event.set()

    def emergency_stop(self) -> None:
        """Bypasses the command queue — see module docstring. Also tears
        down any active motion loop so it can't re-command motion right
        after the :Q#: a live tracking pass re-issues :Rv#+:Me# at 20Hz and
        jog_goto/calibrate at ~7Hz, so writing :Q# alone is overridden
        within one loop tick unless those loops are told to abort too."""
        self._abort.set()
        if self._tracking_stop_event is not None:
            self._tracking_stop_event.set()
        if self._mount is not None:
            self._mount.emergency_stop()

    def shutdown(self) -> None:
        self.stop_tracking()
        self._shutdown.set()
        self._thread.join(timeout=3.0)

    # -- worker thread ------------------------------------------------------

    def _emit(self, kind: str, **payload: Any) -> None:
        self.events.put(WorkerEvent(kind, payload))

    def _run(self) -> None:
        handlers: dict[str, Callable[[dict], None]] = {
            "connect": self._handle_connect,
            "disconnect": self._handle_disconnect,
            "jog_start": self._handle_jog_start,
            "jog_stop": self._handle_jog_stop,
            "stop_all": self._handle_stop_all,
            "park": self._handle_park,
            "park_native": self._handle_park_native,
            "unpark": self._handle_unpark,
            "set_tracking": self._handle_set_tracking,
            "set_altitude_limits": self._handle_set_altitude_limits,
            "goto": self._handle_goto,
            "sync": self._handle_sync,
            "jog_goto": self._handle_jog_goto,
            "calibrate": self._handle_calibrate,
            "measure_mount_lag": self._handle_measure_mount_lag,
            "start_tracking": self._handle_start_tracking,
            "inject_training_pointing_error": self._handle_inject_training_pointing_error,
        }
        # Idle poll cadence for "position" events (~2Hz -- cheap given the
        # real ~5ms round trip measured by characterize.py). Tracked as a
        # timestamp rather than a blocking time.sleep() inside the command
        # loop below -- a sleep() there used to make this the SAME thread
        # that processes jog_start/jog_stop, so a jog click landing mid-
        # sleep sat in _commands for up to 0.4s before the mount even saw
        # it, and the finder/main camera's simulated field (driven by
        # these very "position" events, see App._on_mount_position) lagged
        # real jog motion by however much of that sleep was still pending
        # (confirmed: this is what "the finder field isn't quite in sync
        # with the telescope" during manual jogging traced back to -- not
        # a rendering bug in camera/finder.py at all).
        last_poll_t = 0.0
        while not self._shutdown.is_set():
            try:
                # Short enough that a queued jog command is never stuck
                # behind more than one of these waits -- long enough to not
                # busy-loop when idle.
                name, payload = self._commands.get(timeout=0.05)
            except queue.Empty:
                now = time.monotonic()
                if now - last_poll_t >= IDLE_POLL_INTERVAL_S:
                    self._idle_poll_tick()
                    last_poll_t = now
                continue
            handler = handlers.get(name)
            if handler is None:
                continue
            try:
                handler(payload)
            except Exception as exc:  # noqa: BLE001 - surface it, keep the worker alive
                self._emit("log", message=f"[error] {name} failed: {exc}")

        if self._mount is not None:
            try:
                self._mount.stop()
            except Exception:  # noqa: BLE001 - best effort on shutdown
                pass
            if self._safety is not None:
                self._safety.shutdown()
            self._mount.close()

    def _idle_poll_tick(self) -> None:
        if self._mount is None or not self._idle_poll_enabled.is_set():
            return
        try:
            radec = self._mount.get_radec()
            pier_side = self._mount.get_pier_side()
            self._emit("position", ra_hours=radec.ra_hours, dec_deg=radec.dec_deg, pier_side=pier_side)
        except ProtocolError as exc:
            self._emit("log", message=f"[warn] bad reply during idle poll: {exc}")

    # -- command handlers -----------------------------------------------------

    def _handle_connect(self, payload: dict) -> None:
        kind = payload["kind"]
        transport: Transport
        self._mock_mount = None
        if kind == "mock":
            transport = MockMount(MockConfig(), seed=payload.get("mock_seed"))
            self._mock_mount = transport
        elif kind == "serial":
            transport = SerialTransport(payload["address"], baudrate=9600)
        elif kind == "tcp":
            host, _, port_str = payload["address"].partition(":")
            transport = TCPTransport(host, int(port_str) if port_str else 4030)
        else:
            self._emit("connect_error", message=f"unknown connection kind {kind!r}")
            return

        mount = Mount(transport)
        try:
            firmware = mount.get_version()
            # Without this, :MS# rejects every GOTO with e7# (time/location
            # not synced) -- see Mount.sync_site_and_time's docstring.
            mount.sync_site_and_time(payload["latitude_deg"], payload["longitude_deg"])
        except Exception as exc:  # noqa: BLE001 - report and bail, don't leave a half-open transport
            transport.close()
            self._emit("connect_error", message=str(exc))
            return

        # sync_site_and_time above just pushed THIS machine's clock to the
        # mount (see its docstring) -- the mount has no readable clock of
        # its own to check against afterward, so this is the one point
        # where "is the mount's time actually right" is checkable at all.
        # Soft warning only, never blocks connecting -- same reasoning as
        # the GOTO-mismatch warning below.
        clock_status = check_clock_sync()
        if clock_status.synchronized is False:
            offset = f", offset {clock_status.offset_s:+.2f}s" if clock_status.offset_s is not None else ""
            self._emit("log", message=(
                f"[warn] system clock NOT synchronized ({clock_status.source}{offset}) -- the mount's time was "
                f"just set from this clock, so GOTOs/tracking accuracy may be off by however far it has drifted. "
                f"Fix NTP sync and reconnect."
            ))
        elif clock_status.synchronized is None:
            self._emit("log", message=(
                f"[warn] could not determine system clock sync status ({clock_status.detail}) -- "
                f"the mount's time was just set from this clock regardless."
            ))

        self._mount = mount
        # Runs on this worker thread, not the main thread — can't install
        # signal handlers (see SafetyGuard docstring / am5/safety.py). The
        # GUI's emergency-stop button and the watchdog below are the safety
        # net here instead of Ctrl+C.
        self._safety = SafetyGuard(mount, watchdog_timeout=5.0, install_signal_handlers=False)
        self._connected.set()
        self._idle_poll_enabled.set()
        self._emit("connected", firmware=firmware, connection_kind=kind)

    def _handle_disconnect(self, payload: dict) -> None:
        self._idle_poll_enabled.clear()
        if self._mount is not None:
            self._mount.stop()
            if self._safety is not None:
                self._safety.shutdown()
                self._safety = None
            self._mount.close()
            self._mount = None
        self._mock_mount = None
        self._connected.clear()
        self._parked = False
        self._emit("disconnected")

    def _blocked_while_parked(self, action: str) -> bool:
        """True (and logs a warning) if `action` should be refused because
        the mount is parked. Stopping is never gated this way — only
        commands that start new motion are."""
        if self._parked:
            self._emit("log", message=f"[warn] mount is parked -- unpark before {action}")
            return True
        return False

    def _handle_park(self, payload: dict) -> None:
        if self._mount is None:
            return
        self._mount.stop()
        self._mount.park()
        if self._safety is not None:
            self._safety.notify_command(movement_active=False)
        self._parked = True
        self._emit("parked", method="home", reply=None)

    def _handle_park_native(self, payload: dict) -> None:
        if self._mount is None:
            return
        # Per the official protocol doc, :hP# "only support in equatorial
        # mode" -- the mount does no client-side check of its own.
        status = self._mount.get_status()
        if status.is_equatorial is False:
            self._emit("log", message="[warn] :hP# refused -- mount is in Alt-Az mode, native park needs equatorial mode")
            return
        self._mount.stop()
        reply = self._mount.park_native()
        if self._safety is not None:
            self._safety.notify_command(movement_active=False)
        self._parked = True
        self._emit("parked", method="native", reply=reply.strip())
        # Cross-check against the mount's own P flag rather than trusting
        # only this local software flag.
        after = self._mount.get_status()
        if not after.is_parked:
            self._emit("log", message="[warn] :hP# replied success but :GU# does not report the P flag")

    def _handle_unpark(self, payload: dict) -> None:
        # No wire command -- confirmed true for park() (:hC#, see its
        # docstring). UNVERIFIED for park_native() (:hP#): if the mount
        # still refuses motion after this, :hP# needs a real unpark command
        # we don't know yet -- cross-check the P flag below to find out.
        self._parked = False
        if self._mount is not None:
            status = self._mount.get_status()
            if status.is_parked:
                self._emit("log", message="[warn] :GU# still reports P after unpark -- :hP# likely needs a real wire-level unpark command")
        self._emit("unparked")

    def _handle_jog_start(self, payload: dict) -> None:
        if self._mount is None or self._blocked_while_parked("jogging"):
            return
        self._mount.set_rate(payload["rate_x"])
        self._mount.move(payload["direction"])
        if self._safety is not None:
            self._safety.notify_command(movement_active=True)

    def _handle_jog_stop(self, payload: dict) -> None:
        if self._mount is None:
            return
        self._mount.stop(payload["direction"])
        if self._safety is not None:
            self._safety.notify_command(movement_active=False)

    def _handle_jog_goto(self, payload: dict) -> None:
        if self._mount is None or self._blocked_while_parked("a manual GOTO"):
            return
        target_ra_hours, target_dec_deg = payload["ra_hours"], payload["dec_deg"]
        axis_signs: AxisSigns = payload["axis_signs"]
        self._abort.clear()
        deadline = time.monotonic() + JOG_GOTO_TIMEOUT_S
        arrived = False
        diverged = False
        best_error_deg = float("inf")
        while time.monotonic() < deadline and not self._abort.is_set():
            try:
                radec = self._mount.get_radec()
            except ProtocolError:
                continue
            self._emit("position", ra_hours=radec.ra_hours, dec_deg=radec.dec_deg)
            # NOT auto-correcting axis_signs.dec from a live :Gm# read
            # here anymore -- this was tried and reverted (see AxisSigns'
            # docstring in tracker.py for the full account). Real
            # incident: it fired mid-flight during a real GOTO where no
            # actual mechanical flip should have happened, and the
            # resulting sign flip was the wrong direction, itself causing
            # a real divergence during live ISS tracking. Unresolved
            # whether :Gm# tracks true mechanical pier state during
            # continuous motion (jog/tracking) as opposed to a discrete
            # :MS# GOTO (confirmed correct for that case) -- until that's
            # settled, recalibrate by hand after any deliberate re-point.
            d_ra_deg = circular_diff_hours(radec.ra_hours, target_ra_hours) * 15.0  # actual - target
            d_dec_deg = radec.dec_deg - target_dec_deg
            if abs(d_ra_deg) * 3600.0 < JOG_GOTO_ARRIVED_ARCSEC and abs(d_dec_deg) * 3600.0 < JOG_GOTO_ARRIVED_ARCSEC:
                arrived = True
                break
            # Divergence guard: a wrong axis-sign calibration makes this
            # jog AWAY from the target -- error keeps growing instead of
            # shrinking. Bail out instead of jogging the wrong way for the
            # full 180s timeout. Real great-circle separation, not a
            # tangent-plane hypot(d_ra*cos(dec), d_dec) approximation --
            # that approximation actively lies for a large initial
            # separation (confirmed on real hardware: reported a GROWING
            # error while both raw RA and DEC differences were shrinking,
            # because cos(dec) grows as dec moves away from the pole --
            # see angular_separation_deg's docstring).
            error_deg = angular_separation_deg(radec.ra_hours * 15.0, radec.dec_deg, target_ra_hours * 15.0, target_dec_deg)
            best_error_deg = min(best_error_deg, error_deg)
            if error_deg > best_error_deg + JOG_GOTO_DIVERGENCE_DEG:
                diverged = True
                break
            # Synchronized, not independently clamped: for a large initial
            # separation (outside jog_goto's typical short-final-approach
            # case, but the "GOTO a named star" button doesn't restrict
            # it), independently capping each axis at JOG_GOTO_MAX_RATE_X
            # lets whichever axis has the smaller raw error (e.g. DEC,
            # near the pole) race ahead and finish while the other (e.g.
            # RA, needing 100+ deg) is still saturated -- confirmed on
            # real hardware that this visits a temporarily-WORSE
            # great-circle path than a direct one, tripping the
            # divergence guard above with correct calibration and no
            # pier flip involved. Scaling both by the same factor (so
            # whichever wants the higher rate lands exactly on the cap)
            # keeps their ratio -- and so their estimated time-to-target
            # -- matched, producing a much more direct path.
            ra_rate_uncapped = abs(d_ra_deg) * JOG_GOTO_KP_RATE_X_PER_DEG
            dec_rate_uncapped = abs(d_dec_deg) * JOG_GOTO_KP_RATE_X_PER_DEG
            dominant_rate = max(ra_rate_uncapped, dec_rate_uncapped, 1e-9)
            scale = min(1.0, JOG_GOTO_MAX_RATE_X / dominant_rate)
            ra_rate = max(0.5, ra_rate_uncapped * scale)
            dec_rate = max(0.5, dec_rate_uncapped * scale)
            ra_dir = _pick_direction(-d_ra_deg, axis_signs.ra, "e", "w")
            dec_dir = _pick_direction(-d_dec_deg, axis_signs.dec, "n", "s")
            self._mount.set_rate(ra_rate)
            self._mount.move(ra_dir)
            self._mount.set_rate(dec_rate)
            self._mount.move(dec_dir)
            if self._safety is not None:
                self._safety.notify_command(movement_active=True)  # heartbeat -- watchdog needs this every loop tick, not just once
            time.sleep(0.15)
        self._mount.stop()
        if self._safety is not None:
            self._safety.notify_command(movement_active=False)
        if diverged:
            self._emit("log", message="[warn] Manual GOTO aborted: error grew instead of shrinking -- "
                                      "wrong axis-sign calibration? Run Calibrate and retry.")
        self._emit("jog_goto_result", arrived=arrived)

    def _handle_stop_all(self, payload: dict) -> None:
        if self._mount is None:
            return
        self._mount.stop()
        if self._safety is not None:
            self._safety.notify_command(movement_active=False)

    def _handle_set_tracking(self, payload: dict) -> None:
        if self._mount is None:
            return
        if payload["on"] and self._blocked_while_parked("enabling tracking"):
            return
        self._mount.set_tracking(payload["on"])

    def _handle_set_altitude_limits(self, payload: dict) -> None:
        if self._mount is None:
            return
        self._mount.set_altitude_limits_enabled(payload["enabled"])
        self._emit("log", message=f"altitude limits {'enabled' if payload['enabled'] else 'DISABLED'}")

    def _handle_inject_training_pointing_error(self, payload: dict) -> None:
        if self._mock_mount is None:
            self._emit("log", message="[warn] pointing-error injection requested but the connected mount isn't mock -- ignored")
            return
        self._mock_mount.inject_pointing_error(payload["ra_bias_deg"], payload["dec_bias_deg"])

    def _handle_calibrate(self, payload: dict) -> None:
        if self._mount is None or self._blocked_while_parked("calibrating"):
            return
        self._abort.clear()
        signs = calibrate_directions(self._mount, abort=self._abort)
        self._emit("calibration_done", ra_sign=signs.ra, dec_sign=signs.dec, pier_side=signs.calibrated_pier_side)

    def _handle_measure_mount_lag(self, payload: dict) -> None:
        if self._mount is None or self._blocked_while_parked("measuring mount lag"):
            return
        self._abort.clear()
        result = measure_mount_lag(
            self._mount, rate_x=payload["rate_x"], duration_s=payload["duration_s"], abort=self._abort,
        )
        self._emit(
            "mount_lag_result", lag_s=result.lag_s,
            steady_rate_arcsec_s=result.steady_rate_arcsec_s, samples=result.samples,
        )

    def _handle_goto(self, payload: dict) -> None:
        if self._mount is None or self._blocked_while_parked("a GOTO"):
            return
        ra_hours, dec_deg = payload["ra_hours"], payload["dec_deg"]
        try:
            result = self._mount.goto(ra_hours, dec_deg)
        except ProtocolError as exc:
            self._emit("goto_result", code=None, meaning=str(exc), target_ra_hours=ra_hours, target_dec_deg=dec_deg)
            return
        meaning = MS_REPLY_MEANING.get(result, f"e{-result}" if result < 0 else "undocumented reply code")
        self._emit("goto_result", code=result, meaning=meaning, target_ra_hours=ra_hours, target_dec_deg=dec_deg)
        if result != 0:
            return
        if self._safety is not None:
            self._safety.notify_command(movement_active=True)
        arrived_radec = self._poll_until_arrived()
        if self._safety is not None:
            self._safety.notify_command(movement_active=False)
        if arrived_radec is not None:
            self._check_goto_landed_on_target(ra_hours, dec_deg, arrived_radec)

    def _handle_sync(self, payload: dict) -> None:
        # Not gated by _blocked_while_parked -- sync never moves the mount
        # (see Mount.sync's docstring), only park/jog/goto/calibrate are.
        if self._mount is None:
            return
        ra_hours, dec_deg = payload["ra_hours"], payload["dec_deg"]
        try:
            self._mount.sync(ra_hours, dec_deg)
        except ProtocolError as exc:
            self._emit("sync_result", ok=False, message=str(exc), ra_hours=ra_hours, dec_deg=dec_deg)
            return
        self._emit("sync_result", ok=True, message="Synced", ra_hours=ra_hours, dec_deg=dec_deg)

    def _check_goto_landed_on_target(self, target_ra_hours: float, target_dec_deg: float, arrived: tuple[float, float]) -> None:
        """Cross-checks a "0#, then position stopped changing" GOTO against
        where it actually landed. A real mount that accepted the command
        (not e3#/e5#/e7# etc, which return before this ever runs) and then
        stopped degrees away from the requested target isn't a targeting
        bug on our side -- see GOTO_MISMATCH_WARN_DEG's comment."""
        arrived_ra_hours, arrived_dec_deg = arrived
        ra_err_deg = abs(circular_diff_hours(arrived_ra_hours, target_ra_hours)) * 15.0
        dec_err_deg = abs(arrived_dec_deg - target_dec_deg)
        if ra_err_deg > GOTO_MISMATCH_WARN_DEG or dec_err_deg > GOTO_MISMATCH_WARN_DEG:
            self._emit(
                "log",
                message=(
                    f"[warn] GOTO reported success and settled, but landed "
                    f"{ra_err_deg:.2f} deg RA, {dec_err_deg:.2f} deg DEC off target -- "
                    f"possible stall/obstruction, not a software targeting issue "
                    f"(same target was actually sent). Check :GU# stall flags and "
                    f"physically inspect the mount before retrying."
                ),
            )
            if self._mount is not None:
                try:
                    status = self._mount.get_status()
                    if status.ra_stalled or status.dec_stalled:
                        self._emit("log", message=f"[warn] :GU# reports stall -- RA:{status.ra_stalled} DEC:{status.dec_stalled} (raw: {status.raw!r})")
                except ProtocolError:
                    pass

    def _poll_until_arrived(self) -> tuple[float, float] | None:
        assert self._mount is not None
        deadline = time.monotonic() + GOTO_POLL_TIMEOUT_S
        prev = None
        stable_count = 0
        while time.monotonic() < deadline:
            try:
                radec = self._mount.get_radec()
            except ProtocolError:
                continue
            self._emit("position", ra_hours=radec.ra_hours, dec_deg=radec.dec_deg)
            if prev is not None:
                moved_arcsec = (abs(radec.ra_hours - prev[0]) * 15 + abs(radec.dec_deg - prev[1])) * 3600
                stable_count = stable_count + 1 if moved_arcsec < GOTO_ARRIVED_THRESHOLD_ARCSEC else 0
                if stable_count >= 3:
                    self._emit("goto_arrived", ra_hours=radec.ra_hours, dec_deg=radec.dec_deg)
                    return radec.ra_hours, radec.dec_deg
            prev = (radec.ra_hours, radec.dec_deg)
            time.sleep(0.1)
        self._emit("goto_timeout", timeout_s=GOTO_POLL_TIMEOUT_S)
        return None

    def _handle_start_tracking(self, payload: dict) -> None:
        if self._mount is None or self._safety is None:
            self._emit("tracking_error", message="not connected")
            return
        if self._blocked_while_parked("starting tracking"):
            self._emit("tracking_error", message="mount is parked")
            return
        self._idle_poll_enabled.clear()  # the tracking loop does its own :GMEQ# polling
        self._tracking_stop_event = payload["stop_event"]
        fh = open(payload["csv_path"], "w", newline="")
        try:
            writer = csv_module.DictWriter(fh, fieldnames=TRACKING_CSV_FIELDS)
            writer.writeheader()
            self._emit("tracking_started")
            run_tracking_loop(
                self._mount, self._safety, payload["trajectory"], payload["axis_signs"], payload["offsets"],
                writer, payload["duration_s"], config=payload["config"], stop_event=payload["stop_event"],
                on_tick=lambda tick: self._emit("tracking_tick", **tick),
                # Was print()-only before -- invisible unless the GUI
                # happened to be launched from a terminal with stderr
                # visible. See TrackingConfig.meridian_track_limit_deg's
                # docstring for the real incident this is meant to catch
                # if it ever happens again despite that mitigation.
                on_limit_warning=lambda code: self._emit(
                    "log", message=f"[warn] mount reports :GAT# limit code {code} "
                                    f"({LIMIT_CODE_MEANINGS.get(code, '?')}) -- "
                                    f"tracking may have silently stopped on the mount side",
                ),
            )
        except Exception as exc:  # noqa: BLE001 - report, then always fall through to cleanup below
            self._emit("tracking_error", message=str(exc))
        finally:
            fh.close()
            self._mount.stop()
            self._safety.notify_command(movement_active=False)
            self._tracking_stop_event = None
            self._idle_poll_enabled.set()
            self._emit("tracking_stopped")
