"""Background thread that owns the camera device and serializes every
command sent to it — same rationale and pattern as am5/gui/worker.py's
MountWorker, but for the camera. Deliberately a separate worker/thread:
camera and mount are independent USB devices with no shared resource to
serialize between them, so there's no reason to couple their control loops
(camera runs at 100-200fps, mount polling is ~2-20Hz — very different
cadences that shouldn't share a thread).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from camera.asi_camera import AsiCamera
from camera.fits_writer import write_fits
from camera.mock_camera import MockAsiCamera
from camera.ser_writer import SerWriter

# Commands where only the LATEST queued value matters -- see the read
# loop's own comment on _COALESCE_LATEST_ONLY's use for the real-hardware
# bug this fixes (a UI slider dragged quickly queues many of these).
_COALESCE_LATEST_ONLY = frozenset({"set_exposure_us", "set_gain", "set_roi", "set_bit_depth", "set_sky_context"})

PREVIEW_INTERVAL_S = 0.1  # ~10Hz -- plenty for a framing/focus preview, well under the 100-200fps capture rate
# Added on top of the currently-configured exposure to get the read loop's
# own read_frame() timeout -- covers USB transfer/readout/scheduling
# jitter beyond the pure exposure time itself. See the read loop's own
# comment for the incident this margin (and scaling the timeout with
# exposure at all) fixes.
READ_FRAME_TIMEOUT_MARGIN_MS = 2000
STATS_INTERVAL_S = 1.0
WRITE_BUFFER_TARGET_BYTES = 200 * 1024 * 1024  # ~200MB RAM budget for the SER write-behind buffer
WRITE_BUFFER_MIN_FRAMES = 8
WRITE_BUFFER_MAX_FRAMES = 1000  # cap even for a tiny ROI -- no reason to buffer minutes of frames
# How long _handle_disconnect waits for an active recording's write thread
# to finish before giving up and letting it keep running in the background
# (see _handle_disconnect's own comment) -- module-level so tests can
# shorten it to exercise that path without an actual 10s wait.
DISCONNECT_WRITE_JOIN_TIMEOUT_S = 10.0


@dataclass
class CameraEvent:
    kind: str
    payload: dict = field(default_factory=dict)


#  This project's own real camera's 12-bit ADC range (0-4095) when
# running RAW16 -- same constant am5/gui/panels.py's own
# _scale_16bit_to_8bit_fixed and camera/mock_camera.py's read_frame use.
_CAMERA_ADC_MAX = 4095.0


def frame_to_pgm(frame) -> bytes:
    """Encode a 2D array as a binary PGM (P5) image — tk.PhotoImage reads
    this directly via data=..., no Pillow dependency needed. This is a
    raw-sensor grayscale preview (Bayer mosaic, not debayered), good
    enough for framing/focus, not a colour rendering.

    Regression fix: this used to assume the frame was always uint8
    (frame.tobytes() straight into a hardcoded maxval=255 header) -- a
    RAW16 frame (see AsiCamera's own docstring: the sensor's real 12-bit
    ADC range, 0-4095, packed in uint16, reachable via TransitPanel's own
    bit-depth combo) produced a CORRUPTED preview the moment 16-bit mode
    was used: 2 bytes/pixel actually on the wire, but the header's
    maxval=255 tells every reader (pgm_to_array, tk.PhotoImage) to expect
    1. Scaled to uint8 by a FIXED factor (this sensor's real ADC range,
    not the frame's own per-frame max -- same fixed-vs-adaptive
    reasoning as am5/gui/panels.py's _scale_16bit_to_8bit_fixed, which
    exists for exactly this reason: an adaptive per-frame stretch here
    would silently cancel out real gain changes) BEFORE encoding, so the
    wire format -- and everything downstream: pgm_to_array, blob
    detection, the GUI's own manual "Preview stretch" multiplier -- stays
    uint8 always, unchanged."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame.astype(np.float32) * (255.0 / _CAMERA_ADC_MAX), 0, 255).astype(np.uint8)
    height, width = frame.shape
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    return header + frame.tobytes()


def pgm_to_array(pgm: bytes) -> np.ndarray:
    """Inverse of frame_to_pgm -- decodes a binary PGM (P5) back into a 2D
    uint8 array, e.g. for camera/guiding.py's blob detection to run on a
    "preview_frame" event's payload without needing a second, un-throttled
    frame path from the camera.

    Regression fix, found by code audit: this used to split on the FIRST
    occurrence of the maxval line's literal bytes (partition(b"255\\n")),
    which can also occur INSIDE the dimensions line itself whenever the
    frame's width or height is exactly 255 (e.g. "P5\\n640 255\\n255\\n" --
    partition matches the height's own "255\\n" first), corrupting the
    header/body split (header.split() then yields 2 tokens instead of 3,
    raising ValueError, or worse silently misparsing for other digit
    sequences ending in "255"). frame_to_pgm's header format is fixed at
    exactly 3 newline-terminated lines (magic, dimensions, maxval) before
    the raw pixel body -- find the newline that ends the THIRD line
    directly instead of searching for the maxval value's own bytes, so
    this works for any width/height."""
    first_nl = pgm.index(b"\n")
    second_nl = pgm.index(b"\n", first_nl + 1)
    third_nl = pgm.index(b"\n", second_nl + 1)
    header, body = pgm[:second_nl], pgm[third_nl + 1:]  # header = magic + dimensions lines only; maxval line is skipped (fixed at 255, never parsed)
    _, width_s, height_s = header.split()
    width, height = int(width_s), int(height_s)
    return np.frombuffer(body, dtype=np.uint8).reshape(height, width)


class CameraWorker:
    def __init__(self) -> None:
        self.events: "queue.Queue[CameraEvent]" = queue.Queue()
        self._commands: "queue.Queue[tuple[str, dict]]" = queue.Queue()
        self._camera: AsiCamera | MockAsiCamera | None = None
        # Tracked here (not just forwarded to the camera) so the read
        # loop's own read_frame() timeout can scale with it -- see
        # _handle_set_exposure_us and the read loop's own comment for the
        # regression this fixes. Default matches CameraControlVars' own
        # 1000us default.
        self._exposure_us = 1000
        self._streaming = threading.Event()
        self._ser_writer: SerWriter | None = None
        # Write-behind buffer: the disk write (add_frame -- disk I/O, can
        # stall on a slow/contended disk) happens on its own thread, fed by
        # this bounded queue, instead of inline in the frame-read loop
        # below -- so a slow disk delays the SER file, not the next
        # capture_video_frame() call (which would otherwise show up as
        # sensor-side dropped_frames, exactly the metric this is meant to
        # protect). Owned exclusively by the write thread once recording
        # starts; the main loop only ever puts onto the queue, never
        # touches self._ser_writer directly while a write thread is alive.
        self._write_queue: "queue.Queue[tuple] | None" = None
        self._write_thread: threading.Thread | None = None
        # Created fresh per recording in _handle_start_recording, NOT
        # reused across recordings -- a single shared Event here used to
        # let a write thread that outlived _handle_stop_recording/
        # _handle_disconnect's own join timeout (a wedged/very slow disk)
        # get its stop signal silently cleared by a LATER recording's
        # own stop, since both would .set()/.clear() the same object. A
        # private Event per thread means an orphaned thread's stop signal
        # can never be touched by anything but that thread's own caller.
        self._write_stop_event: threading.Event | None = None
        self._buffer_dropped_frames = 0
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- public, thread-safe API ------------------------------------------------

    def connect(
        self, kind: str, camera_id: int = 0, sdk_path: str | None = None, mock_seed: int | None = None,
        plate_scale_arcsec_per_px: float | None = None, bit_depth: int = 8,
        mock_sensor_width: int = 640, mock_sensor_height: int = 480,
    ) -> None:
        self._commands.put(("connect", {
            "kind": kind, "camera_id": camera_id, "sdk_path": sdk_path, "mock_seed": mock_seed,
            "plate_scale_arcsec_per_px": plate_scale_arcsec_per_px, "bit_depth": bit_depth,
            "mock_sensor_width": mock_sensor_width, "mock_sensor_height": mock_sensor_height,
        }))

    def disconnect(self) -> None:
        self._commands.put(("disconnect", {}))

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        self._commands.put(("set_roi", {"x": x, "y": y, "width": width, "height": height}))

    def set_bit_depth(self, bit_depth: int) -> None:
        """Switches the live video path between 8-bit (fast, default) and
        16-bit (the sensor's real 12-bit ADC range, roughly 2x the
        bandwidth per frame) -- governs both the preview/SER recording
        AND save_fits_snapshot (which just reads whatever the stream is
        currently producing), a single setting for both, not a per-action
        choice."""
        self._commands.put(("set_bit_depth", {"bit_depth": bit_depth}))

    def set_exposure_us(self, microseconds: int) -> None:
        self._commands.put(("set_exposure_us", {"microseconds": microseconds}))

    def set_gain(self, gain: int) -> None:
        self._commands.put(("set_gain", {"gain": gain}))

    def set_sky_context(self, boresight_ra_deg: float, boresight_dec_deg: float, target_ra_deg: float, target_dec_deg: float) -> None:
        """Mount's actual RA/DEC (boresight) and the real-or-reference
        target RA/DEC (the ISS, or a training reference point), fed by the
        App from MountWorker's position/tracking_tick events -- lets the
        mock camera render a real star field + ISS blob reacting to actual
        mount motion/jogging for training. No-op against a real AsiCamera
        (no such concept on real hardware)."""
        self._commands.put(("set_sky_context", {
            "boresight_ra_deg": boresight_ra_deg, "boresight_dec_deg": boresight_dec_deg,
            "target_ra_deg": target_ra_deg, "target_dec_deg": target_dec_deg,
        }))

    def start_recording(self, path: Path, observer: str = "", instrument: str = "ASI290MC", telescope: str = "") -> None:
        self._commands.put(("start_recording", {
            "path": path, "observer": observer, "instrument": instrument, "telescope": telescope,
        }))

    def stop_recording(self) -> None:
        self._commands.put(("stop_recording", {}))

    def save_fits_snapshot(self, path: Path) -> None:
        self._commands.put(("save_fits_snapshot", {"path": path}))

    def shutdown(self) -> None:
        self._shutdown.set()
        self._thread.join(timeout=3.0)

    # -- worker thread ------------------------------------------------------

    def _emit(self, kind: str, **payload: Any) -> None:
        self.events.put(CameraEvent(kind, payload))

    def _run(self) -> None:
        handlers: dict[str, Callable[[dict], None]] = {
            "connect": self._handle_connect,
            "disconnect": self._handle_disconnect,
            "set_roi": self._handle_set_roi,
            "set_bit_depth": self._handle_set_bit_depth,
            "set_exposure_us": self._handle_set_exposure_us,
            "set_gain": self._handle_set_gain,
            "set_sky_context": self._handle_set_sky_context,
            "start_recording": self._handle_start_recording,
            "stop_recording": self._handle_stop_recording,
            "save_fits_snapshot": self._handle_save_fits_snapshot,
        }
        last_preview = 0.0
        last_stats = time.monotonic()  # not 0.0: that reads as falsy below and the first fps report would be a fake 0.0
        frame_count_since_stats = 0
        read_errors_since_stats = 0
        while not self._shutdown.is_set():
            # Regression fix: this used to dequeue exactly ONE command per
            # loop iteration, with a full blocking read_frame() (at
            # whatever exposure was JUST set) after every single one. A UI
            # slider dragged quickly queues many set_exposure_us commands
            # near-instantly -- confirmed live: dragging the finder's
            # exposure slider from 5s down to 50ms looked like it "never"
            # sped back up, because the worker was working through each
            # queued intermediate value one at a time, blocking a full
            # (still slow) exposure's worth on every one of them before
            # even looking at the next command, let alone the final
            # desired value. Reproduced deterministically against the
            # mock camera (not real-hardware-specific): queuing 12 rapid
            # set_exposure_us calls from 5s down to 50ms took ~18s to
            # settle instead of ~1 frame. Fix: drain the WHOLE backlog
            # currently in the queue up front, and for commands where only
            # the latest value actually matters (_COALESCE_LATEST_ONLY),
            # keep just the last one of each -- one-shot commands
            # (connect/disconnect/start_recording/stop_recording/
            # save_fits_snapshot) are still applied in full, in their
            # original order, since dropping or reordering any of those
            # would be a real correctness bug, not just a perf one.
            pending: list[tuple[str, dict]] = []
            while True:
                try:
                    pending.append(self._commands.get_nowait())
                except queue.Empty:
                    break
            # Keep every command in its ORIGINAL relative position (a
            # coalesced command could otherwise jump ahead of an earlier
            # "connect", or behind a later "disconnect" -- either would be
            # a real correctness bug) -- just drop a coalescable command
            # if a newer one of the same name is still coming later in
            # this same batch.
            #
            # Regression fix: dropping was previously keyed only on "is a
            # newer same-name command anywhere later in this batch",
            # ignoring what's BETWEEN the two -- a one-shot command (e.g.
            # start_recording) landing between two set_roi calls changes
            # guard state (_handle_set_roi refuses while recording is
            # active), so silently dropping the EARLIER set_roi lost a
            # real operator action, and the later one then got refused
            # too (both changes gone), contradicting this loop's own
            # promise that one-shot commands are never reordered/dropped
            # relative to coalesced ones. Fix: a one-shot command LOCKS IN
            # every coalescible command seen so far in this batch -- none
            # of them may be dropped once a one-shot command has been
            # seen after them, only a later occurrence of the SAME name
            # with no one-shot command in between may still coalesce it
            # away.
            keep = [True] * len(pending)
            last_index_by_name: dict[str, int] = {}
            for i, (name, _payload) in enumerate(pending):
                if name in _COALESCE_LATEST_ONLY:
                    if name in last_index_by_name:
                        keep[last_index_by_name[name]] = False
                    last_index_by_name[name] = i
                else:
                    last_index_by_name.clear()
            to_run = [(name, payload) for i, (name, payload) in enumerate(pending) if keep[i]]
            for name, payload in to_run:
                handler = handlers.get(name)
                if handler is not None:
                    try:
                        handler(payload)
                    except Exception as exc:  # noqa: BLE001 - surface it, keep the worker alive
                        self._emit("log", message=f"[error] {name} failed: {exc}")

            if self._camera is None or not self._streaming.is_set():
                time.sleep(0.05)
                continue

            try:
                # Regression fix: this used to be a flat 2000ms regardless
                # of the actually-configured exposure -- the finder's own
                # exposure slider goes up to 5s (MAX_FINDER_EXPOSURE_US,
                # camera/finder.py), and a frame simply cannot be ready
                # before its own exposure time elapses, so ANY exposure
                # above ~2s made every single read_frame() call time out
                # (confirmed: reported as a wall of "[finder:log] [warn]
                # read_frame failed: Timeout" with a real 5s exposure/gain
                # 400 configured -- a working camera, not a hardware
                # fault). Scales with the currently-set exposure plus a
                # flat margin for USB transfer/readout/scheduling jitter,
                # floored at the original 2000ms for the common short-
                # exposure case.
                timeout_ms = max(2000, round(self._exposure_us / 1000) + READ_FRAME_TIMEOUT_MARGIN_MS)
                frame = self._camera.read_frame(timeout_ms=timeout_ms)
            except Exception as exc:  # noqa: BLE001 - a dropped/timed-out frame shouldn't kill the loop
                self._emit("log", message=f"[warn] read_frame failed: {exc}")
                read_errors_since_stats += 1
                continue

            frame_count_since_stats += 1
            now_utc = datetime.now(timezone.utc)
            write_queue = self._write_queue
            # Sampled *before* this frame's own put below -- the backlog
            # left over from all prior frames, i.e. the real steady-state
            # occupancy. Sampling right after the put instead would almost
            # always read >=1 (the frame just enqueued, still sitting
            # there because the write thread hasn't been scheduled by the
            # GIL yet to pop it) even when the writer is comfortably
            # keeping up -- a measurement artifact, not a real backlog.
            buffer_used = write_queue.qsize() if write_queue is not None else 0
            buffer_capacity = write_queue.maxsize if write_queue is not None else 0
            if write_queue is not None:
                try:
                    write_queue.put_nowait((frame, now_utc))
                except queue.Full:
                    # The write thread genuinely can't keep up with
                    # sustained disk I/O -- drop the new frame rather than
                    # block here (blocking would just turn this into the
                    # same capture-side stall the buffer exists to avoid).
                    self._buffer_dropped_frames += 1

            now = time.monotonic()
            if now - last_preview >= PREVIEW_INTERVAL_S:
                last_preview = now
                # Buffer occupancy piggybacks on the preview tick (~10Hz)
                # rather than the 1Hz stats tick -- a fill/empty bar at 1Hz
                # would look like it's jumping, not actually filling.
                self._emit(
                    "preview_frame", pgm=frame_to_pgm(frame), width=frame.shape[1], height=frame.shape[0],
                    buffer_used=buffer_used, buffer_capacity=buffer_capacity,
                )
            if now - last_stats >= STATS_INTERVAL_S:
                fps = frame_count_since_stats / (now - last_stats)
                # dropped_frames: the camera's own ring-buffer counter --
                # frames the sensor produced but we didn't fetch in time
                # (bandwidth/exposure mismatch). read_errors: our own
                # read_frame() calls that raised/timed out outright -- a
                # different symptom (comm dropout, USB hiccup), not
                # reported by the SDK's counter (confirmed on real
                # hardware: induced read timeouts left get_dropped_frames()
                # at 0, while starving the read loop for 2s produced 326).
                # buffer_dropped_frames: the write-behind queue was full --
                # a slow disk, not a comm/sensor problem.
                self._emit("stats", fps=fps, recording=self._ser_writer is not None,
                            frames_recorded=self._ser_writer.frame_count if self._ser_writer else 0,
                            file_bytes=self._ser_writer.bytes_written if self._ser_writer else 0,
                            dropped_frames=self._camera.get_dropped_frames(),
                            read_errors=read_errors_since_stats,
                            buffer_dropped_frames=self._buffer_dropped_frames)
                last_stats = now
                frame_count_since_stats = 0
                read_errors_since_stats = 0

        self._handle_disconnect({})

    def _write_loop(self, writer: SerWriter, write_queue: "queue.Queue[tuple]", stop_event: threading.Event) -> None:
        """Runs on its own thread once recording starts -- pulls
        (frame, timestamp) pairs off write_queue and does the actual disk
        write, so a slow disk stalls this thread, not the frame-read loop
        in _run. Polls with a short timeout rather than blocking on
        write_queue.get() forever, so stop_event (set by
        _handle_stop_recording/_handle_disconnect) is noticed promptly;
        once set, drains whatever's left in the queue (in order,
        non-blocking -- self._write_queue is nulled out by whoever signals
        the stop before this loop's caller can enqueue anything new)
        before closing the writer.

        Wrapped in try/except/finally: unlike every command handler in
        _run (each wrapped individually), this thread had no error
        handling at all -- an add_frame()/close() failure (disk full,
        permissions, a truncation bug) used to kill the thread mid-write,
        leaving the SER file with FrameCount=0 and no trailer despite
        real frame bytes already on disk (close() is what patches both).
        Now: on any exception, still close() so whatever was captured
        stays readable.

        Emits "recording_stopped" itself, rather than leaving that to
        whichever command handler signaled the stop: a wedged/very slow
        disk can keep this thread alive well past _handle_stop_recording/
        _handle_disconnect's own join timeout, and reporting "recording
        stopped" from the command handler in that case used to just be
        wrong -- it read self._ser_writer.frame_count from a writer this
        thread might still be mid-write on, before close() ever ran,
        understating (or, once the command handler gave up and moved on,
        entirely fabricating) the real result. Confirmed directly: with
        add_frame() forced to hang past a shortened join timeout,
        stop_recording() reported frame_count=0/error=None while the file
        on disk already held real frame bytes and its header FrameCount
        stayed unpatched. This thread is the only place that actually
        knows when the file is truly finalized, so it's the only place
        that should report it -- whether that's milliseconds or, on a
        stalled disk, much later than the handler that requested the stop
        ever waited around to see."""
        error: Exception | None = None
        try:
            while not stop_event.is_set():
                try:
                    frame, timestamp = write_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                writer.add_frame(frame, timestamp=timestamp)
            while True:
                try:
                    frame, timestamp = write_queue.get_nowait()
                except queue.Empty:
                    break
                writer.add_frame(frame, timestamp=timestamp)
        except Exception as exc:  # noqa: BLE001
            error = exc
        finally:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001
                if error is None:
                    error = exc
        if error is not None:
            self._emit("log", message=f"[error] SER write failed, recording stopped early: {error}")
        self._emit(
            "recording_stopped", path=str(writer.path), frame_count=writer.frame_count,
            buffer_dropped_frames=self._buffer_dropped_frames, error=str(error) if error is not None else None,
        )

    # -- command handlers -----------------------------------------------------

    def _handle_connect(self, payload: dict) -> None:
        kind = payload["kind"]
        bit_depth = payload.get("bit_depth", 8)
        if kind == "mock":
            mock_kwargs = {"seed": payload.get("mock_seed"), "bit_depth": bit_depth}
            if payload.get("plate_scale_arcsec_per_px") is not None:
                mock_kwargs["plate_scale_arcsec_per_px"] = payload["plate_scale_arcsec_per_px"]
            mock_kwargs["sensor_width"] = payload.get("mock_sensor_width", 640)
            mock_kwargs["sensor_height"] = payload.get("mock_sensor_height", 480)
            camera = MockAsiCamera(**mock_kwargs)
        else:
            camera = AsiCamera(payload["camera_id"], payload.get("sdk_path"), bit_depth=bit_depth)
        try:
            camera.open()
        except Exception as exc:  # noqa: BLE001
            self._emit("connect_error", message=str(exc))
            return
        self._camera = camera
        self._camera.start_streaming()
        self._streaming.set()
        self._emit("connected", width=camera.width, height=camera.height, is_color=camera.is_color,
                    controls=camera.get_controls(), bit_depth=camera.bit_depth,
                    colour_id=camera.bayer_pattern_ser_colour_id())

    def _handle_disconnect(self, payload: dict) -> None:
        self._streaming.clear()
        if self._ser_writer is not None:
            # Best-effort: wait for THIS recording's write thread to drain
            # and close so the file on disk is finalized before
            # "disconnected" fires (tested behavior -- a real operator
            # expects the file to be immediately usable right after
            # disconnecting). Bounded, not indefinite -- see _write_loop's
            # own docstring for why a wedged disk shouldn't hang this
            # forever. If it times out, the thread is left running (not
            # nulled here) so a later reconnect+start_recording still
            # correctly refuses until it actually finishes -- see
            # _handle_start_recording's own guard -- and it reports its
            # own "recording_stopped" whenever it's actually done.
            self._write_stop_event.set()
            self._write_queue = None
            self._ser_writer = None
            self._write_thread.join(timeout=DISCONNECT_WRITE_JOIN_TIMEOUT_S)
            if self._write_thread.is_alive():
                self._emit("log", message="[warn] disconnecting while the SER write thread is still finishing (disk busy) -- it will keep writing in the background and report its own result once done")
            else:
                self._write_thread = None
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        self._emit("disconnected")

    def _handle_set_roi(self, payload: dict) -> None:
        if self._camera is None:
            return
        if self._ser_writer is not None:
            # Refused, not silently allowed: the in-progress SerWriter was
            # constructed with the OLD width/height, so the very next frame
            # after a live ROI change has a shape that no longer matches --
            # SerWriter.add_frame() raises on that (by design, to catch
            # exactly this), which used to kill the write thread silently
            # (no error surfaced, writer.close() never ran, so the file's
            # FrameCount header stayed 0 and stop_recording() still reported
            # a clean success) -- confirmed by reproducing it directly.
            self._emit("log", message="[warn] ROI change refused while recording is active -- stop recording first")
            return
        self._camera.set_roi(payload["x"], payload["y"], payload["width"], payload["height"])

    def _handle_set_bit_depth(self, payload: dict) -> None:
        if self._camera is None:
            return
        if self._ser_writer is not None:
            # Same rationale as _handle_set_roi's own guard, different
            # failure mode: SerWriter.add_frame() doesn't validate bit
            # depth, it just casts every frame to the pixel dtype fixed at
            # recording-start time -- a live switch to 16-bit gets silently
            # truncated to 8-bit (numpy's low-byte-only downcast, not a
            # rescale) with no exception and no error anywhere, corrupting
            # pixel data with zero indication anything went wrong
            # (confirmed: a mid-range value of 2048 silently became 0).
            self._emit("log", message="[warn] Bit depth change refused while recording is active -- stop recording first")
            return
        self._camera.set_bit_depth(payload["bit_depth"])
        self._emit("bit_depth_changed", bit_depth=self._camera.bit_depth)

    def _handle_set_exposure_us(self, payload: dict) -> None:
        # Tracked regardless of whether a camera is connected yet (a
        # connect right after this should already use it), same as the
        # read loop reading self._exposure_us on every iteration rather
        # than only right after a successful set.
        self._exposure_us = payload["microseconds"]
        if self._camera is None:
            return
        self._camera.set_exposure_us(payload["microseconds"])

    def _handle_set_gain(self, payload: dict) -> None:
        if self._camera is None:
            return
        self._camera.set_gain(payload["gain"])

    def _handle_set_sky_context(self, payload: dict) -> None:
        if self._camera is None or not hasattr(self._camera, "set_sky_context"):
            return
        self._camera.set_sky_context(
            payload["boresight_ra_deg"], payload["boresight_dec_deg"],
            payload["target_ra_deg"], payload["target_dec_deg"],
        )

    def _handle_start_recording(self, payload: dict) -> None:
        if self._camera is None:
            self._emit("log", message="[error] can't start recording: not connected")
            return
        if self._write_thread is not None and self._write_thread.is_alive():
            # Refused, not blocked: a previous recording's write thread is
            # still draining/closing (a slow disk, or the join-timeout
            # case documented in _handle_disconnect/_write_loop) -- used
            # to block here waiting for it, which is exactly the kind of
            # stall a real operator starting a new recording mid-pass
            # can't afford. It'll finish and report itself; try again.
            self._emit("log", message="[warn] can't start recording: previous recording is still finishing (disk busy) -- try again shortly")
            return
        path = Path(payload["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ser_writer = SerWriter(
            path, width=self._camera.width, height=self._camera.height,
            colour_id=self._camera.bayer_pattern_ser_colour_id(), pixel_depth=self._camera.bit_depth,
            observer=payload.get("observer", ""), instrument=payload.get("instrument", ""),
            telescope=payload.get("telescope", ""),
        )
        # Queue sized off a fixed RAM budget rather than a fixed frame
        # count -- a 16-bit full-sensor frame is ~20x the bytes of a small
        # 8-bit ROI, so a frame-count cap alone would either waste RAM at
        # small ROIs or barely buffer anything at full resolution.
        bytes_per_frame = self._camera.width * self._camera.height * (2 if self._camera.bit_depth == 16 else 1)
        queue_len = max(WRITE_BUFFER_MIN_FRAMES, min(WRITE_BUFFER_MAX_FRAMES, WRITE_BUFFER_TARGET_BYTES // max(bytes_per_frame, 1)))
        self._write_queue = queue.Queue(maxsize=queue_len)
        self._buffer_dropped_frames = 0
        # Fresh per-recording Event -- see __init__'s own comment on why
        # this must never be reused/shared across recordings.
        self._write_stop_event = threading.Event()
        self._write_thread = threading.Thread(
            target=self._write_loop, args=(self._ser_writer, self._write_queue, self._write_stop_event), daemon=True,
        )
        self._write_thread.start()
        self._emit("recording_started", path=str(path))

    def _handle_stop_recording(self, payload: dict) -> None:
        if self._ser_writer is None:
            return
        # Signal and hand off ownership to the write thread -- it drains
        # the queue, closes the writer, and emits "recording_stopped"
        # itself once that's actually done (see _write_loop's own
        # docstring for why this handler doesn't wait for it or report
        # the result itself anymore). Nulling these now (not waiting)
        # marks the session as over for the main capture loop's own stats
        # and for CameraPanel's ROI/bit-depth guard immediately, which is
        # correct regardless of how long the write thread itself takes to
        # actually finish closing the file.
        self._write_stop_event.set()
        self._write_queue = None
        self._ser_writer = None

    def _handle_save_fits_snapshot(self, payload: dict) -> None:
        if self._camera is None:
            self._emit("log", message="[error] can't save snapshot: not connected")
            return
        # Same bit depth as whatever the live video path is currently
        # running at (set_bit_depth) -- one setting governs both, no
        # separate per-action choice. Timeout scales with exposure for
        # the same reason the main read loop's own does -- see its
        # comment for the incident this fixes.
        frame = self._camera.read_frame(
            timeout_ms=max(5000, round(self._exposure_us / 1000) + READ_FRAME_TIMEOUT_MARGIN_MS)
        )
        path = Path(payload["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_fits(path, frame, header_extra={"INSTRUME": "ASI290MC"})
        self._emit("fits_saved", path=str(path), bit_depth=self._camera.bit_depth)
