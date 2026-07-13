from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from camera.ser_reader import SerReader, from_dotnet_ticks
from camera.ser_writer import SerWriter, to_dotnet_ticks


def test_from_dotnet_ticks_is_exact_inverse_of_to_dotnet_ticks():
    dt = datetime(2026, 7, 13, 13, 14, 34, 123456, tzinfo=timezone.utc)
    assert from_dotnet_ticks(to_dotnet_ticks(dt)) == dt


def test_reads_back_header_fields(tmp_path):
    path = tmp_path / "capture.ser"
    with SerWriter(path, width=16, height=12, colour_id=8, pixel_depth=8,
                    observer="tanguy", instrument="ASI290MC", telescope="AM5") as writer:
        writer.add_frame(np.zeros((12, 16), dtype=np.uint8))

    with SerReader(path) as reader:
        h = reader.header
        assert h.width == 16
        assert h.height == 12
        assert h.colour_id == 8
        assert h.colour_name == "BAYER_RGGB"
        assert h.pixel_depth == 8
        assert h.frame_count == 1
        assert h.observer == "tanguy"
        assert h.instrument == "ASI290MC"
        assert h.telescope == "AM5"


def test_reads_back_8bit_frames_byte_identical(tmp_path):
    path = tmp_path / "capture8.ser"
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 256, size=(10, 14), dtype=np.uint8) for _ in range(5)]
    with SerWriter(path, width=14, height=10, colour_id=0) as writer:
        for frame in frames:
            writer.add_frame(frame)

    with SerReader(path) as reader:
        assert reader.frame_count == 5
        for i, expected in enumerate(frames):
            np.testing.assert_array_equal(reader.read_frame(i), expected)


def test_reads_back_16bit_frames_byte_identical(tmp_path):
    path = tmp_path / "capture16.ser"
    rng = np.random.default_rng(2)
    frames = [rng.integers(0, 4096, size=(6, 8), dtype=np.uint16) for _ in range(3)]
    with SerWriter(path, width=8, height=6, colour_id=0, pixel_depth=16) as writer:
        for frame in frames:
            writer.add_frame(frame)

    with SerReader(path) as reader:
        assert reader.header.pixel_depth == 16
        for i, expected in enumerate(frames):
            got = reader.read_frame(i)
            assert got.dtype == np.uint16
            np.testing.assert_array_equal(got, expected)


def test_reads_back_timestamps_in_order(tmp_path):
    path = tmp_path / "capture_ts.ser"
    t0 = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
    timestamps = [t0 + timedelta(milliseconds=100 * i) for i in range(4)]
    with SerWriter(path, width=4, height=4, colour_id=0) as writer:
        for i, ts in enumerate(timestamps):
            writer.add_frame(np.full((4, 4), i, dtype=np.uint8), timestamp=ts)

    with SerReader(path) as reader:
        assert reader.timestamps == timestamps


def test_missing_trailer_reads_as_none_but_frames_still_work(tmp_path):
    # Simulates a truncated/interrupted recording -- header + frame data
    # present, but the file was cut before the optional timestamp trailer.
    path = tmp_path / "no_trailer.ser"
    with SerWriter(path, width=4, height=4, colour_id=0) as writer:
        writer.add_frame(np.full((4, 4), 7, dtype=np.uint8))
        writer.add_frame(np.full((4, 4), 9, dtype=np.uint8))

    from camera.ser_writer import HEADER_SIZE
    frames_end = HEADER_SIZE + 2 * 4 * 4
    with open(path, "r+b") as fh:
        fh.truncate(frames_end)  # chop off the trailer

    with SerReader(path) as reader:
        assert reader.timestamps is None
        assert reader.frame_count == 2
        np.testing.assert_array_equal(reader.read_frame(0), np.full((4, 4), 7, dtype=np.uint8))
        np.testing.assert_array_equal(reader.read_frame(1), np.full((4, 4), 9, dtype=np.uint8))


def test_read_frame_out_of_range_raises(tmp_path):
    path = tmp_path / "capture_oob.ser"
    with SerWriter(path, width=4, height=4, colour_id=0) as writer:
        writer.add_frame(np.zeros((4, 4), dtype=np.uint8))

    with SerReader(path) as reader:
        with pytest.raises(IndexError):
            reader.read_frame(1)
        with pytest.raises(IndexError):
            reader.read_frame(-1)


def test_rejects_a_non_ser_file(tmp_path):
    path = tmp_path / "not_a_ser.ser"
    path.write_bytes(b"not a real SER file at all, just some bytes" * 10)
    with pytest.raises(ValueError):
        SerReader(path)
