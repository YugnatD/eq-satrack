import struct
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from camera.ser_writer import (
    FILE_ID,
    HEADER_SIZE,
    SerWriter,
    to_dotnet_ticks,
)


def test_to_dotnet_ticks_known_value():
    # 0001-01-01 00:00:00 UTC -> tick 0 by definition
    assert to_dotnet_ticks(datetime(1, 1, 1, tzinfo=timezone.utc)) == 0
    # one second later -> exactly 10_000_000 ticks
    assert to_dotnet_ticks(datetime(1, 1, 1, 0, 0, 1, tzinfo=timezone.utc)) == 10_000_000
    # one tick = 100ns = 0.1 microsecond; a 10-microsecond delta is 100 ticks
    t0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(microseconds=10)
    assert to_dotnet_ticks(t1) - to_dotnet_ticks(t0) == 100


def _read_raw_header(path) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read(HEADER_SIZE)
    assert len(raw) == HEADER_SIZE
    file_id = raw[:14]
    (
        lu_id, colour_id, little_endian, width, height, pixel_depth, frame_count,
        observer, instrument, telescope, date_time, date_time_utc,
    ) = struct.unpack("<7i40s40s40sqq", raw[14:])
    return {
        "file_id": file_id, "lu_id": lu_id, "colour_id": colour_id, "little_endian": little_endian,
        "width": width, "height": height, "pixel_depth": pixel_depth, "frame_count": frame_count,
        "observer": observer, "instrument": instrument, "telescope": telescope,
        "date_time": date_time, "date_time_utc": date_time_utc,
    }


def test_header_size_is_178_bytes():
    assert HEADER_SIZE == 178
    assert FILE_ID == b"LUCAM-RECORDER"


def test_round_trip_header_and_pixels(tmp_path):
    path = tmp_path / "test.ser"
    width, height = 16, 10
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, size=(height, width), dtype=np.uint8) for _ in range(5)]

    writer = SerWriter(path, width=width, height=height, colour_id=8, pixel_depth=8,
                        observer="tanguy", instrument="ASI290MC", telescope="Quattro 250P")
    base_time = datetime(2026, 7, 10, 20, 0, 0, tzinfo=timezone.utc)
    for i, frame in enumerate(frames):
        writer.add_frame(frame, timestamp=base_time + timedelta(milliseconds=5 * i))
    writer.close()

    header = _read_raw_header(path)
    assert header["file_id"] == FILE_ID
    assert header["colour_id"] == 8
    assert header["little_endian"] == 0
    assert header["width"] == width
    assert header["height"] == height
    assert header["pixel_depth"] == 8
    assert header["frame_count"] == 5  # patched correctly on close()
    assert header["observer"].decode().strip() == "tanguy"
    assert header["instrument"].decode().strip() == "ASI290MC"
    assert header["telescope"].decode().strip() == "Quattro 250P"
    assert header["date_time"] == header["date_time_utc"]  # no local/UTC distinction tracked

    with open(path, "rb") as fh:
        fh.seek(HEADER_SIZE)
        frame_bytes = width * height
        for expected in frames:
            actual = np.frombuffer(fh.read(frame_bytes), dtype=np.uint8).reshape(height, width)
            assert np.array_equal(actual, expected)

        trailer = fh.read()
    assert len(trailer) == 8 * len(frames)
    timestamps = struct.unpack(f"<{len(frames)}Q", trailer)
    assert list(timestamps) == sorted(timestamps)  # monotonically non-decreasing
    assert timestamps[0] == to_dotnet_ticks(base_time)
    assert timestamps[-1] == to_dotnet_ticks(base_time + timedelta(milliseconds=20))


def test_bytes_written_tracks_header_plus_frames_written_so_far(tmp_path):
    path = tmp_path / "size.ser"
    writer = SerWriter(path, width=10, height=10, colour_id=0)  # 8-bit -- 100 bytes/frame
    assert writer.bytes_written == HEADER_SIZE
    writer.add_frame(np.zeros((10, 10), dtype=np.uint8))
    assert writer.bytes_written == HEADER_SIZE + 100
    writer.add_frame(np.zeros((10, 10), dtype=np.uint8))
    assert writer.bytes_written == HEADER_SIZE + 200
    writer.close()


def test_bytes_written_matches_actual_file_size_before_the_trailer(tmp_path):
    # close() appends the timestamp trailer (8 bytes/frame) after the last
    # add_frame -- bytes_written deliberately excludes it (see its
    # docstring), so it should match the real on-disk size while still
    # recording, and undercount by frame_count*8 right after close().
    import os

    path = tmp_path / "size2.ser"
    writer = SerWriter(path, width=4, height=4, colour_id=0)
    writer.add_frame(np.zeros((4, 4), dtype=np.uint8))
    writer.add_frame(np.zeros((4, 4), dtype=np.uint8))
    writer._fh.flush()
    assert writer.bytes_written == os.path.getsize(path)
    writer.close()
    assert os.path.getsize(path) == writer.bytes_written + 2 * 8


def test_add_frame_rejects_wrong_shape(tmp_path):
    writer = SerWriter(tmp_path / "bad.ser", width=10, height=10, colour_id=0)
    with pytest.raises(ValueError):
        writer.add_frame(np.zeros((5, 5), dtype=np.uint8))
    writer.close()


def test_context_manager_closes_and_patches_frame_count(tmp_path):
    path = tmp_path / "ctx.ser"
    with SerWriter(path, width=4, height=4, colour_id=0) as writer:
        writer.add_frame(np.zeros((4, 4), dtype=np.uint8))
        writer.add_frame(np.ones((4, 4), dtype=np.uint8))
    assert _read_raw_header(path)["frame_count"] == 2


def test_16bit_frames_round_trip(tmp_path):
    path = tmp_path / "16bit.ser"
    width, height = 8, 6
    frame = np.arange(width * height, dtype=np.uint16).reshape(height, width) * 257
    with SerWriter(path, width=width, height=height, colour_id=0, pixel_depth=16) as writer:
        writer.add_frame(frame)

    with open(path, "rb") as fh:
        fh.seek(HEADER_SIZE)
        actual = np.frombuffer(fh.read(width * height * 2), dtype="<u2").reshape(height, width)
    assert np.array_equal(actual, frame)
