"""Reads SER video files back -- the structural inverse of
camera/ser_writer.py. Round-trip tested in tests/test_ser_reader.py against
files SerWriter itself produces, the same "read back what we wrote"
verification approach used to validate ser_writer.py's own hand-rolled
binary format (no independent SER library exists to check against).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from camera.ser_writer import (
    _DOTNET_EPOCH,
    _HEADER_STRUCT,
    _TICKS_PER_SECOND,
    FILE_ID,
    HEADER_SIZE,
)

COLOUR_ID_NAMES = {
    0: "MONO", 8: "BAYER_RGGB", 9: "BAYER_GRBG", 10: "BAYER_GBRG", 11: "BAYER_BGGR",
    100: "RGB", 101: "BGR",
}


def from_dotnet_ticks(ticks: int) -> datetime:
    """Inverse of ser_writer.to_dotnet_ticks -- exact, no rounding: ticks
    is always seconds*1e7 + microseconds*10 (both terms already multiples
    of 10), so dividing the remainder by 10 loses nothing."""
    seconds, remainder_ticks = divmod(ticks, _TICKS_PER_SECOND)
    return _DOTNET_EPOCH + timedelta(seconds=seconds, microseconds=remainder_ticks // 10)


@dataclass
class SerHeader:
    colour_id: int
    width: int
    height: int
    pixel_depth: int
    frame_count: int
    observer: str
    instrument: str
    telescope: str
    date_time: datetime
    date_time_utc: datetime

    @property
    def colour_name(self) -> str:
        return COLOUR_ID_NAMES.get(self.colour_id, f"unknown({self.colour_id})")


class SerReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        self.header = self._read_header()
        self._pixel_dtype = np.dtype("<u1" if self.header.pixel_depth <= 8 else "<u2")
        self._frame_bytes = self.header.width * self.header.height * self._pixel_dtype.itemsize
        self.timestamps = self._read_trailer()

    def _read_header(self) -> SerHeader:
        self._fh.seek(0)
        file_id = self._fh.read(len(FILE_ID))
        if file_id != FILE_ID:
            raise ValueError(f"not a SER file (bad FileID {file_id!r}, expected {FILE_ID!r})")
        (
            _lu_id, colour_id, _little_endian, width, height, pixel_depth, frame_count,
            observer, instrument, telescope, date_time, date_time_utc,
        ) = _HEADER_STRUCT.unpack(self._fh.read(_HEADER_STRUCT.size))
        return SerHeader(
            colour_id=colour_id, width=width, height=height, pixel_depth=pixel_depth,
            frame_count=frame_count,
            observer=observer.decode("ascii", errors="replace").rstrip(),
            instrument=instrument.decode("ascii", errors="replace").rstrip(),
            telescope=telescope.decode("ascii", errors="replace").rstrip(),
            date_time=from_dotnet_ticks(date_time), date_time_utc=from_dotnet_ticks(date_time_utc),
        )

    def _read_trailer(self) -> list[datetime] | None:
        # The timestamp trailer is optional per the SER spec -- absent if
        # the file was truncated/interrupted, or written by a tool that
        # skips it. Detect it from the actual file size rather than
        # assuming it's always there.
        frames_end = HEADER_SIZE + self.header.frame_count * self._frame_bytes
        trailer_size = self.header.frame_count * 8
        if self.header.frame_count == 0 or self.path.stat().st_size < frames_end + trailer_size:
            return None
        self._fh.seek(frames_end)
        raw = self._fh.read(trailer_size)
        ticks = struct.unpack(f"<{self.header.frame_count}Q", raw)
        return [from_dotnet_ticks(t) for t in ticks]

    @property
    def frame_count(self) -> int:
        return self.header.frame_count

    def read_frame(self, index: int) -> np.ndarray:
        if not (0 <= index < self.header.frame_count):
            raise IndexError(f"frame index {index} out of range (0..{self.header.frame_count - 1})")
        self._fh.seek(HEADER_SIZE + index * self._frame_bytes)
        raw = self._fh.read(self._frame_bytes)
        if len(raw) != self._frame_bytes:
            raise OSError(f"short read for frame {index}: got {len(raw)} bytes, expected {self._frame_bytes}")
        return np.frombuffer(raw, dtype=self._pixel_dtype).reshape(self.header.height, self.header.width)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "SerReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
