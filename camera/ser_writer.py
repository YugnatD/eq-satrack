"""SER video format writer — the sequence format used by AutoStakkert!,
PIPP, FireCapture etc. for planetary/lucky-imaging capture.

Verified against the ser-player project's actual source (pipp_ser.h/.cpp,
github.com/cgarry/ser-player), not a summarized PDF, after protocol.py's
earlier mount-side lesson about trusting transcribed docs over the real
implementation. Layout: a 178-byte header (14-byte "LUCAM-RECORDER" FileID
+ 7 int32 fields + 3x40-byte strings + 2 int64 .NET-tick timestamps),
followed by raw frame bytes back to back, followed by an optional trailer
of one 8-byte .NET-tick timestamp per frame.

Known quirk, deliberately handled: the header's LittleEndian field has a
long-standing compatibility bug across the SER ecosystem (documented at
free-astro.org/index.php/SER) where popular tools disagree on its meaning.
The de-facto convention that AutoStakkert!/PIPP actually rely on is
LittleEndian=0 with pixel bytes always written little-endian regardless —
that's what this writer does.
"""

from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FILE_ID = b"LUCAM-RECORDER"
HEADER_SIZE = 178
_HEADER_STRUCT = struct.Struct("<7i40s40s40sqq")
_FRAME_COUNT_OFFSET = len(FILE_ID) + 6 * 4  # FileID + 6 int32 fields precede FrameCount

# SER ColourID values, from pipp_ser.h.
COLOURID_MONO = 0
COLOURID_BAYER_RGGB = 8
COLOURID_BAYER_GRBG = 9
COLOURID_BAYER_GBRG = 10
COLOURID_BAYER_BGGR = 11
COLOURID_RGB = 100
COLOURID_BGR = 101

_TICKS_PER_SECOND = 10_000_000
_DOTNET_EPOCH = datetime(1, 1, 1, tzinfo=timezone.utc)


def to_dotnet_ticks(dt: datetime) -> int:
    """.NET DateTime.Ticks: 100ns intervals since 0001-01-01, proleptic
    Gregorian calendar — same calendar Python's datetime uses, so no
    conversion needed beyond the epoch offset. Integer arithmetic
    throughout to avoid float rounding at 100ns resolution."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = dt.astimezone(timezone.utc) - _DOTNET_EPOCH
    return (delta.days * 86400 + delta.seconds) * _TICKS_PER_SECOND + delta.microseconds * 10


def _pad(text: str, size: int) -> bytes:
    data = text.encode("ascii", errors="replace")[:size]
    return data + b" " * (size - len(data))


class SerWriter:
    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        colour_id: int,
        pixel_depth: int = 8,
        observer: str = "",
        instrument: str = "",
        telescope: str = "",
    ):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.colour_id = colour_id
        self.pixel_depth = pixel_depth
        self._pixel_dtype = np.dtype("<u1" if pixel_depth <= 8 else "<u2")
        self._frame_count = 0
        self._timestamps: list[int] = []
        self._fh = open(self.path, "wb")
        self._write_header(observer, instrument, telescope)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _write_header(self, observer: str, instrument: str, telescope: str) -> None:
        now_ticks = to_dotnet_ticks(datetime.now(timezone.utc))
        self._fh.write(FILE_ID)
        self._fh.write(
            _HEADER_STRUCT.pack(
                0,  # LuID, unused
                self.colour_id,
                0,  # LittleEndian -- see module docstring
                self.width,
                self.height,
                self.pixel_depth,
                0,  # FrameCount, patched in close()
                _pad(observer, 40),
                _pad(instrument, 40),
                _pad(telescope, 40),
                now_ticks,
                now_ticks,
            )
        )

    def add_frame(self, frame: np.ndarray, timestamp: datetime | None = None) -> None:
        if frame.shape != (self.height, self.width):
            raise ValueError(f"frame shape {frame.shape} != ({self.height}, {self.width})")
        self._fh.write(frame.astype(self._pixel_dtype, copy=False).tobytes())
        self._timestamps.append(to_dotnet_ticks(timestamp or datetime.now(timezone.utc)))
        self._frame_count += 1

    def close(self) -> None:
        for ticks in self._timestamps:
            self._fh.write(struct.pack("<Q", ticks))
        self._fh.seek(_FRAME_COUNT_OFFSET)
        self._fh.write(struct.pack("<i", self._frame_count))
        self._fh.close()

    def __enter__(self) -> "SerWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
