"""In-process mock of the AM5's serial protocol, for developing/testing
`characterize.py` without hardware.

The mock is deliberately configurable on the two open questions the real
hardware must answer:

- `rv_mode`: "global" (a single :Rv register shared by both axes, read
  continuously) vs "per_axis" (the driver's own hypothesis: :Rv is latched
  into the axis's velocity the instant :Me/:Mw/:Mn/:Ms is received).
- `tracking_adds`: whether a manual :Me/:Mw/:Mn/:Ms motion adds on top of
  sidereal compensation (net RA/DEC drift == commanded rate regardless of
  tracking state) or replaces it (turns tracking off effectively, so with
  tracking otherwise on you also lose the sidereal cancellation).

This lets `characterize.py`'s analysis functions be exercised against known
ground truth before ever touching the mount.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import protocol
from .angles import equatorial_to_altaz, gmst_deg
from .constants import SIDEREAL_DEG_PER_S
from .transport import Transport

_AXIS_OF_DIR = {"e": "ra", "w": "ra", "n": "dec", "s": "dec"}
# Fixed regardless of simulated pier side, unlike real hardware's DEC axis
# -- see the ":Gm#" dispatch handler below for why that's deliberate.
_SIGN_OF_DIR = {"e": 1.0, "w": -1.0, "n": 1.0, "s": -1.0}


@dataclass
class MockConfig:
    rv_mode: str = "per_axis"  # "per_axis" or "global" — the question this whole rig exists to answer
    tracking_adds: bool = True  # True: :Me adds to sidereal. False: :Me replaces it.
    tau_s: float = 0.15  # mechanical ramp time constant toward commanded velocity
    latency_profile: str = "serial"  # "serial" or "tcp"
    latitude_deg: float = 46.18
    longitude_deg: float = 6.14
    firmware_version: str = "1.8.8"
    start_ra_deg: float = 45.0
    start_dec_deg: float = 45.0
    # Meridian-crossing behavior this mock can simulate -- see
    # protocol.build_set_meridian_behavior's docstring for the real
    # command this simulates (:ST#/:GTa#), never previously wired into
    # this project. meridian_limit_enabled DEFAULTS OFF: _check_meridian_
    # limit computes hour angle from the REAL wall-clock time, so ANY mock
    # session running long enough, or jogged/GOTO'd far enough, can end up
    # sitting at a hour angle that trips it purely by coincidence of what
    # time of day the test happens to run relative to start_ra_deg/
    # longitude_deg -- confirmed TWICE: first broke an unrelated mount-lag
    # test (a static, unmoving start_ra_deg happened to already read as
    # "just past the meridian" for the real time it ran), then broke an
    # unrelated large-jog_goto test (a big ~114 deg RA slew genuinely
    # crossed the meridian mid-maneuver and got frozen, even with the
    # then-"permissive" 15 deg default -- 15 deg isn't a large margin for
    # a real GOTO-sized jog). Given that history, this whole simulation is
    # opt-in: only tests that specifically want to exercise the meridian
    # feature construct their own MockConfig(meridian_limit_enabled=True,
    # ...) -- see test_mount.py -- rather than it running unattended for
    # every other mock user in the codebase.
    meridian_limit_enabled: bool = False
    meridian_track_past: bool = True
    meridian_limit_deg: float = 15.0


@dataclass
class _AxisState:
    moving_dir: str | None = None  # e.g. 'e'/'w' for ra, 'n'/'s' for dec, or None
    vel_target: float = 0.0  # deg/s, signed
    vel_actual: float = 0.0  # deg/s, signed (ramps toward target with tau_s)
    latched_rate: float = 1.0  # x sidereal, used only in per_axis mode


@dataclass
class _MountState:
    ra_deg: float
    dec_deg: float
    tracking: bool = True
    rv_global: float = 1.0
    alt_limits_enabled: bool = True
    alt_low_deg: float = 0.0
    alt_high_deg: float = 90.0
    equatorial_mode: bool = True
    axes: dict = field(default_factory=lambda: {"ra": _AxisState(), "dec": _AxisState()})
    pending_target_ra_hours: float | None = None  # staged by :Sr#, consumed by :MS#/:CM#
    pending_target_dec_deg: float | None = None  # staged by :Sd#
    at_home: bool = False  # set by :hC#, cleared by any subsequent motion
    parked: bool = False  # set by :hP#, cleared by any subsequent motion
    have_datetime: bool = False  # set by :SMTI#
    have_latitude: bool = False  # set by :St#
    have_longitude: bool = False  # set by :Sg#
    meridian_limit_enabled: bool = False  # opt-in, see MockConfig's own docstring for why
    meridian_flip: bool = False  # :ST's first digit -- "temporarily not supported" per the doc, kept for completeness
    # :ST's second digit and sign+2-digit angle -- these defaults match
    # MockConfig's own ones purely so a hypothetical direct _MountState(...)
    # construction isn't silently unsafe; MockMount.__init__ always seeds
    # these explicitly from self._cfg instead of relying on these field
    # defaults.
    meridian_track_past: bool = True
    meridian_limit_deg: float = 15.0
    # True once RA has hit the configured meridian limit while tracking --
    # see _check_meridian_limit. Mirrors the real mount silently refusing
    # further :Rv#/:Me#/:Mw# commands once this happens (see
    # protocol.build_set_meridian_behavior's docstring).
    meridian_stopped: bool = False

    @property
    def site_time_synced(self) -> bool:
        return self.have_datetime and self.have_latitude and self.have_longitude


class MockMount(Transport):
    """Implements the `Transport` interface; drop-in replacement for
    `SerialTransport`/`TCPTransport` in `Mount`."""

    def __init__(self, config: MockConfig | None = None, seed: int | None = None):
        self._cfg = config or MockConfig()
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._state = _MountState(
            ra_deg=self._cfg.start_ra_deg, dec_deg=self._cfg.start_dec_deg,
            meridian_limit_enabled=self._cfg.meridian_limit_enabled,
            meridian_track_past=self._cfg.meridian_track_past, meridian_limit_deg=self._cfg.meridian_limit_deg,
        )
        self._resp_queue: "queue.Queue[bytes]" = queue.Queue()
        self._timers: list[threading.Timer] = []
        self._stop = threading.Event()
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()

    # -- Transport interface -------------------------------------------------

    def write(self, data: bytes) -> None:
        cmd = data.decode("ascii", errors="replace")
        response, has_response = self._dispatch(cmd)
        if not has_response:
            return
        delay = self._sample_latency()
        timer = threading.Timer(delay, self._resp_queue.put, args=(response.encode("ascii"),))
        timer.daemon = True
        with self._lock:
            self._timers.append(timer)
        timer.start()

    def read_until_hash(self, timeout: float) -> bytes:
        try:
            return self._resp_queue.get(timeout=timeout)
        except queue.Empty:
            return b""

    def read_exact(self, n: int, timeout: float) -> bytes:
        # The mock only ever queues whole, pre-formed replies (never streams
        # individual bytes), so this is the same pop as read_until_hash —
        # real :Sr#/:Sd# replies just happen to be 1 byte with no '#'.
        try:
            return self._resp_queue.get(timeout=timeout)
        except queue.Empty:
            return b""

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            for t in self._timers:
                t.cancel()
        self._sim_thread.join(timeout=1.0)

    # -- training-only extension (no wire-protocol equivalent) ----------------

    def inject_pointing_error(self, ra_bias_deg: float, dec_bias_deg: float) -> None:
        """Nudges the SIMULATED mount's own believed position by a fixed
        bias, as if the last GOTO/sync had landed this far off -- unlike
        everything else in this class, this has no wire-protocol
        equivalent and exists purely so a GUI training scenario (see
        MountWorker.inject_training_pointing_error) can rehearse "the ISS
        isn't quite where the mount thinks" for real, rather than faking
        it in the rendered camera frame -- which would silently distort
        FinderCalibration/blob-detection math instead of exercising it.
        Applied directly to state, not via vel_target -- an instant jump,
        not a slew, deliberately: a real residual pointing error is
        already "there" the moment tracking starts, not something that
        visibly arrives via motion."""
        with self._lock:
            self._state.ra_deg = (self._state.ra_deg + ra_bias_deg) % 360.0
            self._state.dec_deg = max(-90.0, min(90.0, self._state.dec_deg + dec_bias_deg))

    # -- physics simulation ---------------------------------------------------

    def _sim_loop(self) -> None:
        target_dt = 0.005
        last_t = time.monotonic()
        while not self._stop.is_set():
            t0 = time.monotonic()
            # Measured, not assumed: under real thread contention (many
            # real commands in flight plus this loop's own response
            # timers), this loop can't always sustain 200Hz. Integrating
            # position with a fixed dt=0.005 regardless of how much wall
            # time actually passed silently under-integrates whenever a
            # cycle runs slow, understating the mount's real position by a
            # growing amount the longer a test/session runs -- confirmed:
            # a several-second run_tracking_loop test against this mock
            # showed a steadily growing (not settling) simulated lag
            # before this fix, on this machine, under the thread load a
            # real tracking-loop-plus-command-issuing test creates.
            real_dt = t0 - last_t
            last_t = t0
            with self._lock:
                self._step(real_dt)
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, target_dt - elapsed))

    def _step(self, dt: float) -> None:
        s = self._state
        for axis_name, axis in s.axes.items():
            alpha = min(dt / self._cfg.tau_s, 1.0)
            axis.vel_actual += (axis.vel_target - axis.vel_actual) * alpha

        sidereal = SIDEREAL_DEG_PER_S if s.tracking else 0.0
        ra_axis, dec_axis = s.axes["ra"], s.axes["dec"]

        if s.meridian_stopped:
            # The mount has silently stopped responding to RA rate commands
            # (see _check_meridian_limit) -- zero the axis every step so it
            # can't drift even if run_tracking_loop keeps pushing :Rv#/:Me#/
            # :Mw#, exactly like the real incident this simulates.
            ra_axis.vel_actual = 0.0
            ra_axis.vel_target = 0.0
        elif self._cfg.tracking_adds:
            # The firmware always silently cancels sidereal drift out of the
            # reported coordinate, whether or not a manual rate is also
            # active, so reported RA only moves by the commanded amount —
            # tracking on/off makes no difference while :Me# is in effect.
            s.ra_deg = (s.ra_deg + ra_axis.vel_actual * dt) % 360.0
        else:
            # The manual command does not suspend the sidereal bookkeeping:
            # it keeps adding whenever tracking is on, so a manual move
            # shows up on top of it — the two are distinguishable by exactly
            # one sidereal rate's worth of dRA/dt.
            s.ra_deg = (s.ra_deg + (sidereal + ra_axis.vel_actual) * dt) % 360.0
        s.dec_deg = max(-90.0, min(90.0, s.dec_deg + dec_axis.vel_actual * dt))
        self._check_meridian_limit(s)

    def _check_meridian_limit(self, s: _MountState) -> None:
        """Simulates the mount's own :ST-configured meridian-crossing limit
        (see protocol.build_set_meridian_behavior). Hour angle (HA = LST -
        RA) increases with real elapsed time while tracking, exactly like
        the ":Gm#" pier-side computation below uses -- once it crosses the
        configured threshold, the mount silently stops the RA axis (DEC is
        unaffected, matching the doc's own note that only "the RA axis...
        continues to track the angle of rotation" up to the limit) and
        :GAT# starts reporting e8# instead of 1#, reproducing the real
        "tracking diverges right after the meridian" incident this exists
        to catch. Opt-in via MockConfig.meridian_limit_enabled -- see its
        own docstring for why this must not run unattended for mock
        sessions that aren't specifically testing this feature."""
        if not s.meridian_limit_enabled or not s.tracking or s.meridian_stopped:
            return
        lst_deg = (gmst_deg(datetime.now(timezone.utc)) + self._cfg.longitude_deg) % 360.0
        ha_deg = ((lst_deg - s.ra_deg) + 180.0) % 360.0 - 180.0
        # Only meaningful within a bounded window around the meridian --
        # a mount can legitimately sit at ANY hour angle for hours (GOTO'd
        # straight there, or simply tracking a target that's been up a
        # while), which is a normal static state, not "actively crossing
        # the meridian right now". Without this bound, a mock simply
        # constructed/left running long enough for its (fixed, from
        # MockConfig) start_ra_deg to end up at a large HA relative to the
        # real wall-clock LST would falsely trip this regardless of
        # meridian_limit_deg -- confirmed: this broke an unrelated,
        # meridian-unaware test the first time this was tried, purely
        # because that test's default start_ra_deg happened to already sit
        # at HA=+162 deg (nowhere near the meridian) when it ran. Bounded
        # generously above the protocol's own 15 deg max limit so a real
        # configured limit is never mistaken for "not close enough to
        # check".
        if not (-30.0 <= ha_deg <= 30.0):
            return
        # track_past=False: doc's own stated default stop angle, 1 degree
        # past the meridian. track_past=True: the configured limit_deg
        # (which may be negative, meaning "stop before even reaching the
        # meridian" -- see build_set_meridian_behavior's docstring).
        threshold_deg = s.meridian_limit_deg if s.meridian_track_past else 1.0
        if ha_deg >= threshold_deg:
            s.meridian_stopped = True

    def _sample_latency(self) -> float:
        if self._cfg.latency_profile == "tcp":
            base = max(0.001, self._rng.gauss(0.008, 0.015))
            if self._rng.random() < 0.05:
                base += self._rng.uniform(0.05, 0.2)
            return base
        return max(0.005, self._rng.gauss(0.031, 0.005))

    # -- command dispatch -------------------------------------------------

    def _dispatch(self, cmd: str) -> tuple[str, bool]:
        with self._lock:
            s = self._state
            if cmd == ":GMEQ#":
                return f"{protocol.format_ra_hours(s.ra_deg / 15.0)}&{protocol.format_signed_dms(s.dec_deg)}#", True
            if cmd == ":GMZA#":
                az, alt = self._compute_azalt(s.ra_deg, s.dec_deg)
                return f"{protocol.format_unsigned_dms(az)}&{protocol.format_signed_dms(alt)}#", True
            if cmd == ":GV#":
                return f"{self._cfg.firmware_version}#", True
            if cmd == ":GAT#":
                if s.tracking and s.meridian_stopped:
                    return "e8#", True  # see _check_meridian_limit
                return ("1#" if s.tracking else "0#"), True
            if cmd == ":GTa#":
                return f"{protocol.format_meridian_behavior(s.meridian_flip, s.meridian_track_past, s.meridian_limit_deg)}#", True
            if cmd == ":GU#":
                return self._fake_status(s), True
            if cmd == ":GLC#":
                return "1#", True
            if cmd == ":Gm#":
                # NOTE this readout is intentionally NOT coupled to
                # _SIGN_OF_DIR below: real hardware's DEC motor response
                # actually flips sense with pier side (confirmed on real
                # AM3 hardware -- see AxisSigns' docstring in tracker.py),
                # but simulating that here would make DEC's sign depend on
                # the real wall-clock time a test happens to run (since
                # this is computed from real LST), silently flaking every
                # test that jogs DEC from the default start position
                # depending on time of day. Tests that specifically need a
                # pier flip's effect on DEC sign patch Mount.get_pier_side
                # directly instead (see test_gui_worker.py) rather than
                # relying on a real crossing here.
                if s.at_home:
                    return "N#", True
                lst_deg = (gmst_deg(datetime.now(timezone.utc)) + self._cfg.longitude_deg) % 360.0
                ha_deg = ((lst_deg - s.ra_deg) + 180.0) % 360.0 - 180.0
                return ("E#" if ha_deg < 0 else "W#"), True
            if cmd.startswith(":Rv") and cmd.endswith("#"):
                rate = float(cmd[3:-1])
                s.rv_global = rate
                if self._cfg.rv_mode == "global":
                    # The shared register is read continuously by whichever
                    # axes are already moving, not just latched at :M<dir># time.
                    for axis_name, axis in s.axes.items():
                        if axis.moving_dir is not None:
                            axis.vel_target = _SIGN_OF_DIR[axis.moving_dir] * rate * SIDEREAL_DEG_PER_S
                return "", False
            if cmd in (":Me#", ":Mw#", ":Mn#", ":Ms#"):
                direction = cmd[2]
                axis_name = _AXIS_OF_DIR[direction]
                axis = s.axes[axis_name]
                axis.moving_dir = direction
                if self._cfg.rv_mode == "per_axis":
                    axis.latched_rate = s.rv_global
                rate = axis.latched_rate if self._cfg.rv_mode == "per_axis" else s.rv_global
                axis.vel_target = _SIGN_OF_DIR[direction] * rate * SIDEREAL_DEG_PER_S
                s.at_home = False
                s.parked = False
                return "", False
            if cmd in (":Qe#", ":Qw#", ":Qn#", ":Qs#"):
                direction = cmd[2]
                axis = s.axes[_AXIS_OF_DIR[direction]]
                axis.moving_dir = None
                axis.vel_target = 0.0
                return "", False
            if cmd == ":Q#":
                for axis in s.axes.values():
                    axis.moving_dir = None
                    axis.vel_target = 0.0
                return "", False
            if cmd == ":Te#":
                s.tracking = True
                s.meridian_stopped = False  # a fresh "start tracking" gets a fresh chance
                return "1", True  # bare, no '#' -- confirmed against the protocol doc and real hardware
            if cmd == ":Td#":
                s.tracking = False
                return "1", True
            if cmd in (":TQ#", ":TS#", ":TL#"):
                return "", False
            if cmd in (":AP#", ":AA#"):
                s.equatorial_mode = cmd == ":AP#"
                return "", False
            if cmd == ":hC#":
                s.ra_deg, s.dec_deg = self._cfg.start_ra_deg, self._cfg.start_dec_deg
                s.at_home = True
                s.parked = False
                s.meridian_stopped = False  # back at the home position, well clear of any limit
                return "", False
            if cmd == ":hP#":
                # Arbitrary, distinct from :hC#'s home position purely so a
                # --mock session can tell the two apart visually. Not a
                # claim about the real ZWO native park position (unknown —
                # see Mount.park_native()'s docstring). Per the official doc,
                # only supported in equatorial mode.
                if not s.equatorial_mode:
                    return "0#", True
                s.ra_deg, s.dec_deg = 90.0, 0.0
                s.at_home = False
                s.parked = True
                return "1#", True
            if cmd == ":SLD#":
                s.alt_limits_enabled = False
                return "1", True  # bare, no '#' -- confirmed against real hardware
            if cmd == ":SLE#":
                s.alt_limits_enabled = True
                return "1", True
            if cmd.startswith(":SLL") and cmd.endswith("#"):
                val = float(cmd[4:-1])
                if 0 <= val <= 30:
                    s.alt_low_deg = val
                    return "1#", True
                return "e1#", True
            if cmd.startswith(":SLH") and cmd.endswith("#"):
                val = float(cmd[4:-1])
                if 60 <= val <= 90:
                    s.alt_high_deg = val
                    return "1#", True
                return "e1#", True
            if cmd.startswith(":ST") and cmd.endswith("#"):
                try:
                    flip, track_past, limit_deg = protocol.parse_meridian_behavior(cmd[3:])
                except protocol.ProtocolError:
                    return "0", True
                s.meridian_flip = flip
                s.meridian_track_past = track_past
                s.meridian_limit_deg = limit_deg
                s.meridian_stopped = False  # a fresh configuration gets a fresh chance
                return "1", True  # bare, no '#' -- UNCONFIRMED, see Mount.set_meridian_behavior's docstring
            if cmd.startswith(":Sr") and cmd.endswith("#"):
                try:
                    s.pending_target_ra_hours = protocol.parse_ra_hours(cmd[3:-1])
                    return "1", True
                except ValueError:
                    return "0", True
            if cmd.startswith(":Sd") and cmd.endswith("#"):
                try:
                    s.pending_target_dec_deg = protocol.parse_signed_dms(cmd[3:-1])
                    return "1", True
                except ValueError:
                    return "0", True
            if cmd.startswith(":SMTI") and cmd.endswith("#"):
                # Compound date/time/timezone -- mock doesn't need the value
                # itself, just that this step of the sync sequence happened.
                # Bare "1", no '#' -- confirmed against real hardware.
                s.have_datetime = True
                return "1", True
            if cmd.startswith(":St") and cmd.endswith("#"):
                try:
                    protocol.parse_signed_dms(cmd[3:-1])
                except ValueError:
                    return "e2#", True
                s.have_latitude = True
                return "1", True
            if cmd.startswith(":Sg") and cmd.endswith("#"):
                try:
                    protocol.parse_signed_dms(cmd[3:-1])
                except ValueError:
                    return "e2#", True
                s.have_longitude = True
                return "1", True
            if cmd == ":MS#":
                return self._handle_slew(s)
            if cmd == ":CM#":
                return self._handle_sync(s)
            if cmd.startswith(":Mg") and cmd.endswith("#"):
                return "", False
            if cmd.startswith(":Rg") and cmd.endswith("#"):
                return "", False
            return "e2#", True

    def _handle_slew(self, s: _MountState) -> tuple[str, bool]:
        """:MS# — slew to the :Sr#/:Sd#-staged target. '#'-terminated like
        the rest of the protocol, confirmed against real hardware: 0#
        slewing, 1# below horizon, 2# below the altitude limit, e7# if
        :SMTI#/:St#/:Sg# haven't all been sent yet (confirmed against real
        hardware, which rejected every GOTO with e7# until this session
        found the official doc's init sequence). Teleports instantly rather
        than ramping, unlike manual :Me#/:Mn# moves — real GOTO slew
        dynamics are still uncharacterized (see characterize.py test h)."""
        if not s.site_time_synced:
            return "e7#", True
        if s.pending_target_ra_hours is None or s.pending_target_dec_deg is None:
            return "0#", True
        target_ra_deg = s.pending_target_ra_hours * 15.0
        target_dec_deg = s.pending_target_dec_deg
        _, alt = self._compute_azalt(target_ra_deg, target_dec_deg)
        if alt < 0.0:
            return "1#", True
        if s.alt_limits_enabled and not (s.alt_low_deg <= alt <= s.alt_high_deg):
            return "2#", True
        s.ra_deg, s.dec_deg = target_ra_deg, target_dec_deg
        s.at_home = False
        s.parked = False
        s.meridian_stopped = False  # a fresh GOTO lands well clear of any limit
        return "0#", True

    def _handle_sync(self, s: _MountState) -> tuple[str, bool]:
        if s.pending_target_ra_hours is None or s.pending_target_dec_deg is None:
            return "N/A#", True
        s.ra_deg = s.pending_target_ra_hours * 15.0
        s.dec_deg = s.pending_target_dec_deg
        s.meridian_stopped = False  # a fresh sync lands well clear of any limit
        return "N/A#", True

    def _fake_status(self, s: _MountState) -> str:
        # Flag characters per the official protocol doc (see protocol.py's
        # parse_gu_status), which lists them but not a fixed field layout —
        # each is "or not shown", so this just includes whichever apply.
        # The doc doesn't pin down the exact n/N distinction ("no tracking"
        # vs "stop or tracking"); this picks N when tracking is on, n when
        # it's off, as a reasonable best-effort reading.
        chars = ["N" if s.tracking else "n"]
        if s.at_home:
            chars.append("H")
        chars.append("G" if s.equatorial_mode else "Z")
        if s.parked:
            chars.append("P")
        return "".join(chars) + "001000060#"

    def _compute_azalt(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        # Real wall-clock sidereal time, not an arbitrary counter from mock
        # startup -- otherwise a real (correctly-computed) pass target from
        # am5/ephemeris.py would be checked against a fake sky with no
        # relation to the real one, and "above horizon right now" would be
        # basically a coin flip instead of matching reality.
        return equatorial_to_altaz(ra_deg, dec_deg, self._cfg.latitude_deg, self._cfg.longitude_deg, datetime.now(timezone.utc))
