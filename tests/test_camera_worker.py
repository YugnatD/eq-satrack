import queue
import struct
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
    _wait_for(worker, "fits_saved", timeout=5.0)
    assert path.exists()
