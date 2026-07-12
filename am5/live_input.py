"""Non-blocking keyboard control for live Δt / perpendicular offset nudges
during a pass. Stdlib only (termios/tty/select) — Linux/macOS terminals.

A keyboard has no reliable key-held signal, so the perpendicular offset is a
timed pulse per keypress (see LiveOffsets.trigger_perp_pulse), not a
continuous deflection like a joystick would give — that's deferred to a
later iteration per the implementation plan.
"""

from __future__ import annotations

import select
import sys
import termios
import threading
import tty
from typing import Callable

from .tracker import LiveOffsets

KEY_HELP = """
  live keyboard controls:
    [ / ]   delta_t -0.1s / +0.1s
    { / }   delta_t -1.0s / +1.0s
    a / d   perpendicular nudge left / right (tap)
    q       quit the tracking loop early
"""

_POLL_TIMEOUT_S = 0.1


class KeyboardInput:
    """Background thread reading raw keypresses into a shared LiveOffsets.
    Use as a context manager so the terminal mode is always restored, even
    on an exception or emergency stop — SafetyGuard's SIGINT handler raises
    SystemExit in the main thread, which unwinds through this __exit__ same
    as any other exception."""

    def __init__(self, offsets: LiveOffsets, on_quit: Callable[[], None] | None = None):
        self._offsets = offsets
        self._on_quit = on_quit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_settings: list | None = None

    def __enter__(self) -> "KeyboardInput":
        if not sys.stdin.isatty():
            print("[live_input] stdin is not a tty — keyboard control disabled", file=sys.stderr)
            return self
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        print(KEY_HELP, file=sys.stderr)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def _run(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], _POLL_TIMEOUT_S)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if ch:
                self._handle_key(ch)

    def _handle_key(self, ch: str) -> None:
        if ch == "[":
            self._offsets.adjust_delta_t(-0.1)
        elif ch == "]":
            self._offsets.adjust_delta_t(0.1)
        elif ch == "{":
            self._offsets.adjust_delta_t(-1.0)
        elif ch == "}":
            self._offsets.adjust_delta_t(1.0)
        elif ch == "a":
            self._offsets.trigger_perp_pulse(sign=-1.0)
        elif ch == "d":
            self._offsets.trigger_perp_pulse(sign=1.0)
        elif ch == "q" and self._on_quit is not None:
            self._on_quit()


if __name__ == "__main__":
    # Standalone manual test: run this file directly, press keys, watch the
    # offsets change. No mount, no trajectory — just the input plumbing.
    import time

    offsets = LiveOffsets()
    with KeyboardInput(offsets):
        print("press [ ] { } a d, Ctrl+C to quit", file=sys.stderr)
        try:
            while True:
                dt, perp = offsets.snapshot()
                print(f"\rdelta_t={dt:+.2f}s  perp_pulse={perp:+.0f}   ", end="", file=sys.stderr)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print()
