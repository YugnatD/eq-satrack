"""Non-negotiable safety net: emergency stop on signal, watchdog on silence.

Every entry point that moves the mount must construct a SafetyGuard and call
`notify_command(movement_active=...)` on every command sent. If no command
arrives for `watchdog_timeout` seconds while a movement was last reported
active, the guard fires `:Q#` on its own.
"""

from __future__ import annotations

import signal
import sys
import threading
import time

from .mount import Mount

TUBE_REMOVED_PROMPT = (
    "\n"
    "!!! SAFETY CHECK !!!\n"
    "This script drives the mount at up to 6 deg/s (1440x sidereal).\n"
    "The OTA (tube) MUST be removed from the mount before proceeding.\n"
    "An unbalanced or loaded mount slewing at speed is a real hazard to\n"
    "the equipment and to anyone nearby.\n"
    "\n"
    "Type exactly: TUBE REMOVED\n"
    "> "
)


def confirm_tube_removed() -> None:
    """Blocking confirmation gate. Raises SystemExit if not confirmed."""
    answer = input(TUBE_REMOVED_PROMPT)
    if answer.strip() != "TUBE REMOVED":
        print("Confirmation not received verbatim — aborting.", file=sys.stderr)
        raise SystemExit(1)


READY_TO_TRACK_PROMPT = (
    "\n"
    "!!! PRE-PASS CHECK !!!\n"
    "This is a live tracking run — the OTA is expected to be mounted this\n"
    "time. Before continuing, confirm:\n"
    "  - the mount is manually pointed at the pass's starting RA/DEC (printed above)\n"
    "  - the starting pier side avoids a meridian flip during the pass (see above)\n"
    "  - cabling has enough slack for the pass's full range of motion\n"
    "\n"
    "Type exactly: READY TO TRACK\n"
    "> "
)


def confirm_ready_to_track() -> None:
    """Blocking confirmation gate for an actual pass, distinct from
    confirm_tube_removed() — the safety context is different (tube on,
    camera attached, mount already pointed) so the wording must not be
    reused verbatim."""
    answer = input(READY_TO_TRACK_PROMPT)
    if answer.strip() != "READY TO TRACK":
        print("Confirmation not received verbatim — aborting.", file=sys.stderr)
        raise SystemExit(1)


class SafetyGuard:
    def __init__(
        self, mount: Mount, watchdog_timeout: float = 5.0, poll_interval: float = 0.5,
        install_signal_handlers: bool = True,
    ):
        self._mount = mount
        self._watchdog_timeout = watchdog_timeout
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._last_command_time = time.monotonic()
        self._movement_active = False
        self._stop_event = threading.Event()
        self._signal_handlers_installed = install_signal_handlers
        if install_signal_handlers:
            # signal.signal() only works from the main thread — GUI code
            # constructs SafetyGuard on a background worker thread and must
            # pass install_signal_handlers=False; it relies on its own
            # always-visible emergency-stop control instead of Ctrl+C.
            self._prev_sigint = signal.signal(signal.SIGINT, self._handle_signal)
            self._prev_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def notify_command(self, movement_active: bool) -> None:
        with self._lock:
            self._last_command_time = time.monotonic()
            self._movement_active = movement_active

    def _watchdog_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            with self._lock:
                stale = time.monotonic() - self._last_command_time > self._watchdog_timeout
                active = self._movement_active
            if stale and active:
                print("[SAFETY] watchdog timeout with movement active — sending :Q#", file=sys.stderr)
                self._mount.emergency_stop()
                with self._lock:
                    self._movement_active = False

    def _handle_signal(self, signum, frame) -> None:
        print(f"\n[SAFETY] signal {signum} received — emergency stop", file=sys.stderr)
        self._mount.emergency_stop()
        self.shutdown()
        sys.exit(1)

    def shutdown(self) -> None:
        self._stop_event.set()
        self._mount.emergency_stop()
        if self._signal_handlers_installed:
            signal.signal(signal.SIGINT, self._prev_sigint)
            signal.signal(signal.SIGTERM, self._prev_sigterm)
