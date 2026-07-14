"""Shared decode/state for CameraWorker's real preview_frame stream.

One instance is owned by App and fed every camera.worker.CameraEvent once
(see App._pump_events) rather than each panel re-decoding the same PGM
independently -- AlignmentPanel/AcquisitionPanel/FlatsPanel all read the
exact same last_frame/frame_seq off this object from their own existing
periodic redraw ticks, so no new push/callback plumbing is needed on top
of what those panels already have.

frame_seq exists specifically so a capture (see AcquisitionPanel/
FlatsPanel's real-mode _on_capture_*) can tell a genuinely NEW frame
apart from the same one still sitting here between two ticks of its own
faster poll loop -- the camera's preview stream only updates at ~10Hz
(PREVIEW_INTERVAL_S in camera/worker.py), slower than some panels' own
redraw tick.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable

import numpy as np

from camera.worker import CameraEvent, pgm_to_array


class LiveCameraFeed:
    def __init__(self) -> None:
        self.connected = False
        # Set by ConnectionPanel._on_camera_connect right before it issues
        # the connect command (see that method) -- the "connected" event
        # payload itself carries no kind info (CameraWorker's mock backend
        # connects exactly like a real one), so this is the only place
        # that knows whether THIS connection was requested as mock or
        # real. is_active (below) checks this, not just `connected`, so
        # selecting Mock and connecting successfully does NOT flip
        # AlignmentPanel/AcquisitionPanel/FlatsPanel into real-camera mode.
        self.kind: str | None = None
        self.width: int | None = None
        self.height: int | None = None
        self.is_color = False
        self.bit_depth = 8
        self.controls: dict | None = None
        self.last_frame: np.ndarray | None = None
        self.frame_seq = 0
        # Incremented/decremented by RealCaptureState.start()/consume() --
        # lets AcquisitionPanel/FlatsPanel's tab-visibility exposure/gain
        # resync (see their own _live_tick) tell whether pushing THIS
        # tab's own settings to the shared camera would clobber a capture
        # running on a DIFFERENT tab (Reference/Target/Flats all share
        # one physical camera/CameraWorker).
        self.active_capture_count = 0

    @property
    def is_active(self) -> bool:
        """True only for a genuinely connected REAL camera -- the single
        predicate AlignmentPanel/AcquisitionPanel/FlatsPanel all read
        instead of each re-deriving their own "am I real" check (which
        previously checked `connected` alone and could be fooled by a
        connected Mock backend)."""
        return self.connected and self.kind == "real"

    def handle_event(self, event: CameraEvent) -> None:
        if event.kind == "connected":
            self.connected = True
            self.width = event.payload["width"]
            self.height = event.payload["height"]
            self.is_color = event.payload["is_color"]
            self.bit_depth = event.payload.get("bit_depth", 8)
            self.controls = event.payload.get("controls")
            self.last_frame = None
        elif event.kind == "disconnected":
            self.connected = False
            self.last_frame = None
        elif event.kind == "preview_frame":
            self.last_frame = pgm_to_array(event.payload["pgm"])
            self.frame_seq += 1

    def get_control_range(self, name: str, default_min: float, default_max: float) -> tuple[float, float]:
        """(min, max) for a named real control (e.g. "Exposure", "Gain")
        -- falls back to the given defaults if not connected/reported,
        same "not wrong, just least-assuming" spirit as the rest of this
        app's fallbacks."""
        if self.controls is None or name not in self.controls:
            return default_min, default_max
        entry = self.controls[name]
        return float(entry.get("MinValue", default_min)), float(entry.get("MaxValue", default_max))

    def slider_to_exposure_us(self, slider_0_100: float) -> float:
        """Maps a 0-100 UI slider onto a LOG scale between a practical
        exposure range and the real camera's own reported limits -- a
        linear map across the SDK's full range (32us-2000s on the
        ASI290MC) would waste nearly the entire slider on exposures no
        Star Analyser target would ever need, same reasoning real camera-
        control software uses log exposure sliders. Shared by
        AcquisitionPanel and FlatsPanel so both map the same way."""
        lo, hi = self.get_control_range("Exposure", 100.0, 2_000_000_000.0)
        practical_lo, practical_hi = max(lo, 100.0), min(hi, 10_000_000.0)  # 10s ceiling
        t = slider_0_100 / 100.0
        return practical_lo * (practical_hi / practical_lo) ** t

    def slider_to_gain(self, slider_0_100: float) -> int:
        lo, hi = self.get_control_range("Gain", 0.0, 570.0)
        return round(lo + (hi - lo) * (slider_0_100 / 100.0))


class TabResyncTracker:
    """When should a real-camera-consuming tab push its own exposure/gain
    to the shared camera? Used identically by AcquisitionPanel and
    FlatsPanel's own _live_tick, factored out since both needed the exact
    same edge logic (easy to get subtly wrong twice, same reasoning as
    RealCaptureState above): resync when the tab BECOMES the visible one,
    or when the camera BECOMES real while the tab is already visible
    (connecting real hardware while sitting on that tab must not require
    leaving and re-entering it to pick up the right settings) -- but
    never while ANY panel has a capture in flight, since forcing a push
    then would silently overwrite whatever exposure/gain that capture is
    relying on (e.g. the forced minimum exposure a bias/offset capture
    depends on)."""

    def __init__(self) -> None:
        self._was_mapped = False
        self._was_real = False

    def update(self, is_mapped: bool, is_real_now: bool, active_capture_count: int) -> bool:
        should_resync = False
        if is_mapped:
            became_visible = not self._was_mapped
            became_real = is_real_now and not self._was_real
            should_resync = (became_visible or became_real) and is_real_now and active_capture_count == 0
        self._was_mapped = is_mapped
        self._was_real = is_real_now
        return should_resync


class RealCaptureState:
    """Async accumulation of N real camera frames into a target list --
    used by AcquisitionPanel and FlatsPanel's real-mode capture buttons.

    A click calls start(); consume() (called every tick of the owning
    panel's own _live_tick, same cadence as its local-patch redraw)
    appends one NEW frame per tick -- run through the given crop_fn --
    until n are collected, then calls finalize. Frames only arrive from
    the camera at ~10Hz (LiveCameraFeed.frame_seq), unlike the synthetic
    generators elsewhere in this app which produce all n frames instantly
    -- this exists so both panels share the exact same (easy to get
    subtly wrong) "only count a genuinely new frame once" logic rather
    than reimplementing it twice."""

    def __init__(self, live_camera_feed: LiveCameraFeed):
        self._feed = live_camera_feed
        self._target: list[np.ndarray] | None = None
        self._remaining = 0
        self._total = 0
        self._seen_seq = -1
        self._label = ""
        self._status_var: tk.StringVar | None = None
        self._crop_fn: Callable[[np.ndarray], np.ndarray] | None = None
        self._finalize: Callable[[], None] | None = None
        self._on_abort: Callable[[], None] | None = None

    @property
    def active(self) -> bool:
        return self._target is not None

    def start(
        self, target: list[np.ndarray], n: int, status_var: tk.StringVar, label: str,
        crop_fn: Callable[[np.ndarray], np.ndarray], finalize: Callable[[], None],
        on_abort: Callable[[], None] | None = None,
    ) -> bool:
        """Arms a new capture, or refuses if one is already active --
        returns True if it actually started. Without this guard a second
        click (or a different capture button sharing this same instance)
        would silently overwrite _target/_crop_fn/_finalize mid-flight,
        orphaning the first capture's partially-filled list forever with
        its status frozen and no completion callback ever firing.

        `on_abort`, if given, runs INSTEAD of `finalize` if the capture is
        cut short by a mid-capture disconnect (see consume()) -- distinct
        from `finalize` because finalize reports SUCCESS (bumps a frame
        counter, sets a "✅ ... frames" status), which would be wrong to
        do for a capture that didn't actually finish. Callers that force
        a hardware setting for the duration of the capture (e.g. minimum
        exposure for a bias/offset frame) and need it restored either way
        should pass the restore logic as `on_abort` too."""
        if self.active:
            return False
        self._target = target
        self._remaining = n
        self._total = n
        self._status_var = status_var
        self._label = label
        self._crop_fn = crop_fn
        self._finalize = finalize
        self._on_abort = on_abort
        self._seen_seq = self._feed.frame_seq
        status_var.set(f"⏳ Capturing {label}... 0/{n}")
        self._feed.active_capture_count += 1
        return True

    def _reset(self) -> None:
        self._target = None
        self._status_var = None
        self._crop_fn = None
        self._finalize = None
        self._on_abort = None
        self._feed.active_capture_count -= 1

    def consume(self) -> None:
        if self._target is None or self._remaining <= 0:
            return
        if not self._feed.connected:
            # Camera dropped mid-capture -- abort rather than silently
            # hang forever with the status frozen at "k/n" and no way to
            # start a new capture (active would otherwise never clear).
            done = self._total - self._remaining
            status_var, label, total, on_abort = self._status_var, self._label, self._total, self._on_abort
            self._reset()
            status_var.set(f"⚠ Capturing {label} aborted -- camera disconnected ({done}/{total} collected).")
            if on_abort is not None:
                on_abort()
            return
        if self._feed.last_frame is None or self._feed.frame_seq == self._seen_seq:
            return  # no NEW frame since the last tick yet
        self._seen_seq = self._feed.frame_seq
        self._target.append(self._crop_fn(self._feed.last_frame))
        self._remaining -= 1
        done = self._total - self._remaining
        if self._remaining > 0:
            self._status_var.set(f"⏳ Capturing {self._label}... {done}/{self._total}")
            return
        finalize = self._finalize
        self._reset()
        if finalize is not None:
            finalize()
