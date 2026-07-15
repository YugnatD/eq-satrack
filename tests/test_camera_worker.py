import queue
import struct
import threading
import time

import numpy as np
import pytest

from camera.mock_camera import DEFAULT_ARCSEC_PER_PIXEL
from camera.ser_writer import HEADER_SIZE
from camera.worker import CameraEvent, CameraWorker


def _pgm_to_array(pgm: bytes) -> np.ndarray:
    header, _, body = pgm.partition(b"255\n")
    _, width_s, height_s = header.split()
    width, height = int(width_s), int(height_s)
    return np.frombuffer(body, dtype=np.uint8).reshape(height, width)


def _wait_for(worker: CameraWorker, kind: str, timeout: float = 5.0) -> CameraEvent:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = worker.events.get(timeout=0.1)
        except queue.Empty:
            continue
        if event.kind == kind:
            return event
    raise AssertionError(f"never saw a {kind!r} event within {timeout}s")


@pytest.fixture
def worker():
    w = CameraWorker()
    yield w
    w.shutdown()


def test_connect_emits_connected_with_dimensions(worker):
    worker.connect("mock", mock_seed=1)
    event = _wait_for(worker, "connected")
    assert event.payload["width"] == 640
    assert event.payload["height"] == 480
    assert event.payload["is_color"] is True


def test_preview_frames_arrive_after_connect(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    event = _wait_for(worker, "preview_frame", timeout=3.0)
    assert event.payload["pgm"].startswith(b"P5\n")
    assert event.payload["width"] > 0 and event.payload["height"] > 0


def test_stats_report_positive_fps(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_exposure_us(2000)  # keep the mock's frame pacing fast for a quick test
    event = _wait_for(worker, "stats", timeout=3.0)
    assert event.payload["fps"] > 0


def test_stats_report_dropped_frames_and_read_errors_fields(worker):
    # The mock never actually drops (see MockAsiCamera.get_dropped_frames)
    # or errors, but the fields must always be present -- CameraPanel reads
    # them unconditionally to flag comm/bandwidth problems on real hardware.
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_exposure_us(2000)
    event = _wait_for(worker, "stats", timeout=3.0)
    assert event.payload["dropped_frames"] == 0
    assert event.payload["read_errors"] == 0


def test_recording_produces_a_valid_ser_file(worker, tmp_path):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_exposure_us(1000)

    path = tmp_path / "capture.ser"
    worker.start_recording(path, observer="tanguy", instrument="ASI290MC")
    _wait_for(worker, "recording_started")
    time.sleep(0.5)
    worker.stop_recording()
    stopped = _wait_for(worker, "recording_stopped", timeout=3.0)

    assert stopped.payload["frame_count"] > 0
    assert path.exists()

    with open(path, "rb") as fh:
        raw_header = fh.read(HEADER_SIZE)
        frame_count = struct.unpack("<i", raw_header[14 + 6 * 4 : 14 + 7 * 4])[0]
        assert frame_count == stopped.payload["frame_count"]
        remaining = fh.read()
    # remaining bytes = frame data (width*height*frame_count) + 8 bytes/frame trailer
    frame_bytes = 640 * 480
    expected = frame_count * frame_bytes + frame_count * 8
    assert len(remaining) == expected


def test_stop_recording_does_not_hang_when_the_write_buffer_is_full(worker, tmp_path, monkeypatch):
    # Regression test: _stop_write_thread used to signal the write thread
    # via a sentinel pushed through a blocking queue.put(), which could
    # stall the entire camera worker (frame reads AND all further
    # commands) if the write-behind queue was full at exactly the moment
    # recording was stopped -- precisely the slow-disk scenario the
    # buffer exists to protect against, and exactly when an operator is
    # most likely to be clicking Stop. Force a tiny queue and a slow
    # writer to reliably fill it, then confirm stop still completes
    # promptly (signaled via a threading.Event now, not a queued sentinel).
    import camera.worker as worker_module
    from camera.ser_writer import SerWriter

    monkeypatch.setattr(worker_module, "WRITE_BUFFER_MIN_FRAMES", 2)
    monkeypatch.setattr(worker_module, "WRITE_BUFFER_TARGET_BYTES", 1)  # forces the 2-frame floor above

    real_add_frame = SerWriter.add_frame

    def slow_add_frame(self, frame, timestamp=None):
        time.sleep(0.3)
        real_add_frame(self, frame, timestamp=timestamp)

    monkeypatch.setattr(SerWriter, "add_frame", slow_add_frame)

    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_exposure_us(1000)  # fast frame production, easily fills a 2-frame queue

    path = tmp_path / "slow.ser"
    worker.start_recording(path)
    _wait_for(worker, "recording_started")
    time.sleep(0.5)  # let frames pile up past the tiny queue capacity

    t0 = time.monotonic()
    worker.stop_recording()
    stopped = _wait_for(worker, "recording_stopped", timeout=5.0)
    elapsed = time.monotonic() - t0

    assert stopped.payload["frame_count"] > 0
    # Bounded by draining a handful of slow add_frame calls, not stuck
    # waiting indefinitely for queue space the way a blocking put() would.
    assert elapsed < 3.0


def test_recording_stopped_reports_buffer_dropped_frames(worker, tmp_path):
    # The write-behind buffer (see CameraWorker._write_loop) decouples
    # disk I/O from the capture loop -- this field must always be present
    # so CameraPanel can flag a genuinely slow disk, distinct from
    # dropped_frames (sensor-side) and read_errors (comm-side).
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_exposure_us(1000)

    path = tmp_path / "capture2.ser"
    worker.start_recording(path)
    _wait_for(worker, "recording_started")
    time.sleep(0.3)
    worker.stop_recording()
    stopped = _wait_for(worker, "recording_stopped", timeout=3.0)
    assert stopped.payload["buffer_dropped_frames"] == 0


def test_disconnect_while_recording_closes_a_valid_ser_file(worker, tmp_path):
    # Disconnecting mid-recording must still drain the write-behind queue
    # and close the SER file properly (via _stop_write_thread), not leave
    # a truncated/corrupt file or a frame_count mismatched with the header.
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_exposure_us(1000)

    path = tmp_path / "capture_disconnect.ser"
    worker.start_recording(path)
    _wait_for(worker, "recording_started")
    time.sleep(0.3)
    worker.disconnect()
    _wait_for(worker, "disconnected", timeout=5.0)

    with open(path, "rb") as fh:
        raw_header = fh.read(HEADER_SIZE)
        frame_count = struct.unpack("<i", raw_header[14 + 6 * 4 : 14 + 7 * 4])[0]
        remaining = fh.read()
    assert frame_count > 0
    frame_bytes = 640 * 480
    assert len(remaining) == frame_count * frame_bytes + frame_count * 8


def test_roi_change_is_refused_while_recording_and_file_stays_valid(worker, tmp_path):
    # Regression: a live ROI change used to reach the mock camera mid-
    # recording, so the very next frame had a shape mismatching the
    # already-open SerWriter -- add_frame() raised, killing the write
    # thread with no exception handling anywhere: the file was left with
    # FrameCount=0 in its header despite real frame bytes on disk (see
    # CameraWorker._handle_set_roi's own comment). Now the change is
    # refused outright (logged, not applied) while a recording is active,
    # and the resulting file must still be a normal, valid SER file.
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_exposure_us(1000)

    path = tmp_path / "roi_guard.ser"
    worker.start_recording(path)
    _wait_for(worker, "recording_started")
    time.sleep(0.2)

    worker.set_roi(10, 10, 320, 240)
    warning = _wait_for(worker, "log", timeout=2.0)
    assert "ROI change refused" in warning.payload["message"]

    time.sleep(0.2)
    worker.stop_recording()
    stopped = _wait_for(worker, "recording_stopped", timeout=3.0)
    assert stopped.payload["frame_count"] > 0
    assert stopped.payload.get("error") is None

    with open(path, "rb") as fh:
        raw_header = fh.read(HEADER_SIZE)
        frame_count = struct.unpack("<i", raw_header[14 + 6 * 4 : 14 + 7 * 4])[0]
        remaining = fh.read()
    assert frame_count == stopped.payload["frame_count"]
    frame_bytes = 640 * 480  # ROI change was refused -- still full-frame 8-bit
    assert len(remaining) == frame_count * frame_bytes + frame_count * 8


def test_bit_depth_change_is_refused_while_recording_and_pixels_stay_correct(worker, tmp_path):
    # Regression: a live bit-depth switch used to reach the mock camera
    # mid-recording; SerWriter.add_frame() doesn't validate bit depth, it
    # just casts to the dtype fixed at recording-start, silently
    # truncating 16-bit pixel values to 8-bit with no error anywhere (see
    # CameraWorker._handle_set_bit_depth's own comment). Now refused
    # outright while recording is active.
    worker.connect("mock", mock_seed=1, bit_depth=16)
    _wait_for(worker, "connected")
    worker.set_exposure_us(1000)

    path = tmp_path / "bitdepth_guard.ser"
    worker.start_recording(path)
    _wait_for(worker, "recording_started")
    time.sleep(0.2)

    worker.set_bit_depth(8)
    warning = _wait_for(worker, "log", timeout=2.0)
    assert "Bit depth change refused" in warning.payload["message"]

    time.sleep(0.2)
    worker.stop_recording()
    stopped = _wait_for(worker, "recording_stopped", timeout=3.0)
    assert stopped.payload["frame_count"] > 0

    with open(path, "rb") as fh:
        raw_header = fh.read(HEADER_SIZE)
        pixel_depth = struct.unpack("<i", raw_header[14 + 5 * 4 : 14 + 6 * 4])[0]
    assert pixel_depth == 16  # change was refused -- still recording at the depth set at start_recording


def test_write_thread_failure_still_closes_a_readable_ser_file_and_reports_error(worker, tmp_path, monkeypatch):
    # Regression: _write_loop previously had no exception handling at all
    # -- any add_frame() failure (disk full, permissions, or the ROI/
    # bit-depth corruption above if it ever slipped past the new guards)
    # killed the thread before writer.close() ran, leaving FrameCount=0
    # and no trailer despite real frame bytes already on disk. Now the
    # loop always closes the writer and surfaces the failure instead of
    # reporting a clean stop.
    from camera.ser_writer import SerWriter

    real_add_frame = SerWriter.add_frame
    call_count = {"n": 0}

    def flaky_add_frame(self, frame, timestamp=None):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise OSError("simulated disk failure")
        real_add_frame(self, frame, timestamp=timestamp)

    monkeypatch.setattr(SerWriter, "add_frame", flaky_add_frame)

    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_exposure_us(1000)

    path = tmp_path / "flaky.ser"
    worker.start_recording(path)
    _wait_for(worker, "recording_started")
    time.sleep(0.5)  # let the flaky write happen and kill the write thread

    worker.stop_recording()
    stopped = _wait_for(worker, "recording_stopped", timeout=3.0)

    assert stopped.payload.get("error") is not None
    assert "simulated disk failure" in stopped.payload["error"]
    # The 2 frames written before the failure must still be a valid,
    # readable file -- close() ran despite the exception.
    assert stopped.payload["frame_count"] == 2
    with open(path, "rb") as fh:
        raw_header = fh.read(HEADER_SIZE)
        frame_count = struct.unpack("<i", raw_header[14 + 6 * 4 : 14 + 7 * 4])[0]
        remaining = fh.read()
    assert frame_count == 2
    frame_bytes = 640 * 480
    assert len(remaining) == frame_count * frame_bytes + frame_count * 8


def test_start_recording_is_refused_while_a_previous_recordings_write_thread_is_still_draining(worker, tmp_path, monkeypatch):
    # Regression: _handle_start_recording used to block (join with a
    # timeout) waiting for a previous recording's write thread, then
    # proceed either way -- if that join timed out (a genuinely stalled
    # disk), the new recording would start concurrently with an orphaned
    # old write thread, and the two used to share a single reused
    # threading.Event for their stop signal, letting one clear the
    # other's. Now start_recording refuses outright while the previous
    # write thread is still alive, and the old recording finishes and
    # reports itself once it actually can.
    from camera.ser_writer import SerWriter

    real_add_frame = SerWriter.add_frame
    release = threading.Event()

    def blocking_add_frame(self, frame, timestamp=None):
        release.wait(timeout=5.0)
        real_add_frame(self, frame, timestamp=timestamp)

    monkeypatch.setattr(SerWriter, "add_frame", blocking_add_frame)

    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_exposure_us(1000)

    path1 = tmp_path / "first.ser"
    worker.start_recording(path1)
    _wait_for(worker, "recording_started")
    time.sleep(0.2)  # let a frame reach add_frame and block there

    worker.stop_recording()
    time.sleep(0.2)  # the write thread is signaled to stop but still blocked inside add_frame

    path2 = tmp_path / "second.ser"
    worker.start_recording(path2)
    warning = _wait_for(worker, "log", timeout=2.0)
    assert "previous recording is still finishing" in warning.payload["message"]
    assert not path2.exists()

    release.set()  # unblock the stalled write -- the first recording can now actually finish
    stopped = _wait_for(worker, "recording_stopped", timeout=3.0)
    assert stopped.payload["path"] == str(path1)
    assert stopped.payload["frame_count"] > 0
    assert stopped.payload["error"] is None

    # A second recording now succeeds normally.
    worker.start_recording(path2)
    _wait_for(worker, "recording_started")
    time.sleep(0.2)
    worker.stop_recording()
    stopped2 = _wait_for(worker, "recording_stopped", timeout=3.0)
    assert stopped2.payload["path"] == str(path2)
    assert stopped2.payload["frame_count"] > 0


def test_disconnect_with_a_stalled_write_thread_warns_instead_of_reporting_false_success(worker, tmp_path, monkeypatch):
    # Regression, confirmed by direct reproduction before this fix: with a
    # shortened join timeout and add_frame() forced to hang past it,
    # stop_recording() (and disconnect, which shared the same code path)
    # reported recording_stopped with frame_count=0/error=None -- a
    # fabricated clean result -- while the file on disk already held real
    # frame bytes and its header FrameCount stayed unpatched, because the
    # write thread was still alive and hadn't reached close() yet. Now
    # disconnect warns instead of lying, and the write thread reports the
    # real result itself whenever it actually finishes.
    import camera.worker as worker_module
    from camera.ser_writer import SerWriter

    monkeypatch.setattr(worker_module, "DISCONNECT_WRITE_JOIN_TIMEOUT_S", 0.2)

    real_add_frame = SerWriter.add_frame
    release = threading.Event()

    def blocking_add_frame(self, frame, timestamp=None):
        release.wait(timeout=5.0)
        real_add_frame(self, frame, timestamp=timestamp)

    monkeypatch.setattr(SerWriter, "add_frame", blocking_add_frame)

    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_exposure_us(1000)

    path = tmp_path / "stalled.ser"
    worker.start_recording(path)
    _wait_for(worker, "recording_started")
    time.sleep(0.2)  # let the write thread block on add_frame

    worker.disconnect()
    saw_warning = False
    saw_disconnected = False
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not saw_disconnected:
        try:
            event = worker.events.get(timeout=0.1)
        except queue.Empty:
            continue
        if event.kind == "log" and "still finishing" in event.payload.get("message", ""):
            saw_warning = True
        if event.kind == "disconnected":
            saw_disconnected = True
    assert saw_disconnected
    assert saw_warning  # honest about not having actually finalized the file yet

    release.set()  # the stalled write thread can now actually finish, in the background
    stopped = _wait_for(worker, "recording_stopped", timeout=3.0)
    assert stopped.payload["frame_count"] > 0
    assert stopped.payload["error"] is None
    with open(path, "rb") as fh:
        raw_header = fh.read(HEADER_SIZE)
        frame_count = struct.unpack("<i", raw_header[14 + 6 * 4 : 14 + 7 * 4])[0]
    assert frame_count == stopped.payload["frame_count"]  # the eventual report matches what's actually on disk


def test_connect_with_bit_depth_16_is_reflected_in_connected_event(worker):
    worker.connect("mock", mock_seed=1, bit_depth=16)
    event = _wait_for(worker, "connected")
    assert event.payload["bit_depth"] == 16


def test_recording_at_16bit_produces_a_ser_file_with_pixel_depth_16(worker, tmp_path):
    worker.connect("mock", mock_seed=1, bit_depth=16)
    _wait_for(worker, "connected")
    worker.set_exposure_us(1000)

    path = tmp_path / "capture16.ser"
    worker.start_recording(path)
    _wait_for(worker, "recording_started")
    time.sleep(0.5)
    worker.stop_recording()
    stopped = _wait_for(worker, "recording_stopped", timeout=3.0)
    assert stopped.payload["frame_count"] > 0

    with open(path, "rb") as fh:
        raw_header = fh.read(HEADER_SIZE)
        pixel_depth = struct.unpack("<i", raw_header[14 + 5 * 4 : 14 + 6 * 4])[0]
        frame_count = struct.unpack("<i", raw_header[14 + 6 * 4 : 14 + 7 * 4])[0]
        remaining = fh.read()
    assert pixel_depth == 16
    assert frame_count == stopped.payload["frame_count"]
    frame_bytes = 640 * 480 * 2  # uint16 -- 2 bytes/pixel
    expected = frame_count * frame_bytes + frame_count * 8
    assert len(remaining) == expected


def test_set_bit_depth_after_connect_switches_recording_depth(worker, tmp_path):
    # Live switch mid-session, not just at connect time -- mirrors the
    # real AsiCamera.set_bit_depth's stop/restart bracket, confirmed safe
    # on real ASI290MC hardware.
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_bit_depth(16)
    _wait_for(worker, "bit_depth_changed")

    path = tmp_path / "capture_switched.ser"
    worker.start_recording(path)
    _wait_for(worker, "recording_started")
    time.sleep(0.3)
    worker.stop_recording()
    _wait_for(worker, "recording_stopped", timeout=3.0)

    with open(path, "rb") as fh:
        raw_header = fh.read(HEADER_SIZE)
        pixel_depth = struct.unpack("<i", raw_header[14 + 5 * 4 : 14 + 6 * 4])[0]
    assert pixel_depth == 16


def test_set_sky_context_moves_the_preview_blob(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_exposure_us(1000)

    dx_px = 40
    target_ra = 45.0 + (dx_px * DEFAULT_ARCSEC_PER_PIXEL / 3600.0)
    worker.set_sky_context(boresight_ra_deg=45.0, boresight_dec_deg=0.0, target_ra_deg=target_ra, target_dec_deg=0.0)
    # drain a couple of frames so a stale pre-offset preview isn't sampled
    for _ in range(3):
        event = _wait_for(worker, "preview_frame", timeout=3.0)
    frame = _pgm_to_array(event.payload["pgm"]).astype(float)

    _, peak_x = np.unravel_index(np.argmax(frame), frame.shape)
    assert peak_x == pytest.approx(frame.shape[1] / 2 + dx_px, abs=3)


def test_disconnect_emits_disconnected(worker):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.disconnect()
    _wait_for(worker, "disconnected", timeout=3.0)


def test_fits_snapshot_is_saved(worker, tmp_path):
    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    path = tmp_path / "snapshot.fits"
    worker.save_fits_snapshot(path)
    event = _wait_for(worker, "fits_saved", timeout=5.0)
    assert path.exists()
    assert event.payload["bit_depth"] == 8


def test_fits_snapshot_matches_the_currently_set_bit_depth(worker, tmp_path):
    # One setting governs both the video path and the snapshot -- no
    # separate per-action bit depth choice (see set_bit_depth's docstring).
    from astropy.io import fits

    worker.connect("mock", mock_seed=1)
    _wait_for(worker, "connected")
    worker.set_bit_depth(16)
    _wait_for(worker, "bit_depth_changed")

    path = tmp_path / "snapshot_16bit.fits"
    worker.save_fits_snapshot(path)
    event = _wait_for(worker, "fits_saved", timeout=5.0)
    assert event.payload["bit_depth"] == 16

    with fits.open(path) as hdul:
        data = hdul[0].data
        # astropy round-trips uint16 FITS data as int32 with BZERO/BSCALE
        # applied -- assert on value range, not the raw dtype.
        assert data.min() >= 0
        assert data.max() <= 4095
