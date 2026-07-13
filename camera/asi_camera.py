"""Thin wrapper around zwoasi.Camera — the real ZWO ASI camera device.
Mirrors MockAsiCamera's interface (camera/mock_camera.py) by duck typing
(same pattern as Mount/MockMount in am5/) so camera/worker.py can use
either interchangeably.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zwoasi as asi

# zwoasi.ASI_BAYER_* (0..3) -> SER ColourID (see ser_writer.py for the full
# SER colour table). Best-effort, deduced by elimination from the ZWO SDK's
# 4 pattern values against SER's 4 standard Bayer layouts — not verified
# against a real sensor (no camera available while writing this). If a
# recorded SER opens with visibly swapped colours, override the ColourID in
# your stacking software rather than trust this mapping blindly.
_BAYER_TO_SER_COLOUR_ID = {
    0: 8,  # ASI_BAYER_RG -> COLOURID_BAYER_RGGB
    1: 11,  # ASI_BAYER_BG -> COLOURID_BAYER_BGGR
    2: 9,  # ASI_BAYER_GR -> COLOURID_BAYER_GRBG
    3: 10,  # ASI_BAYER_RB -> COLOURID_BAYER_GBRG (odd SDK name, by elimination)
}

_DEFAULT_SDK_PATHS = [
    "/home/tanguy/ASIStudio/lib/libASICamera2.so",
    "/usr/local/lib/libASICamera2.so",
    "/usr/lib/libASICamera2.so",
]


def find_sdk_library(sdk_path: str | None = None) -> str:
    if sdk_path:
        return sdk_path
    for candidate in _DEFAULT_SDK_PATHS:
        if Path(candidate).exists():
            return candidate
    found = asi.find_library("ASICamera2")
    if found:
        return found
    raise RuntimeError("libASICamera2.so not found. Pass sdk_path explicitly or install the ZWO ASI SDK.")


_BIT_DEPTH_TO_IMG_TYPE = {8: asi.ASI_IMG_RAW8, 16: asi.ASI_IMG_RAW16}


class AsiCamera:
    """A real ZWO ASI camera, opened via the vendor SDK (zwoasi). RAW8 by
    default (no 16-bit/colour-debayer conversion) — the right format for
    high framerate planetary/ISS capture where bandwidth is the constraint.
    RAW16 (the sensor's real 12-bit ADC range, 0-4095, packed in uint16) is
    available via set_bit_depth(16) for whoever wants full dynamic range in
    the recorded SER sequence itself, not just a one-off snapshot -- costs
    roughly 2x the USB bandwidth per frame (confirmed on a real ASI290MC:
    fps was actually unaffected at 640x480/5ms, exposure time was already
    the bottleneck there, but a full-sensor or longer-exposure recording
    will be bandwidth-bound and should expect a real fps hit)."""

    def __init__(self, camera_id: int = 0, sdk_path: str | None = None, bit_depth: int = 8):
        self._camera_id = camera_id
        self._sdk_path = sdk_path
        self._camera: asi.Camera | None = None
        self._width = 0
        self._height = 0
        self._is_color = False
        self._bayer_pattern: int | None = None
        self._bit_depth = bit_depth
        self._streaming = False

    def open(self) -> None:
        asi.init(find_sdk_library(self._sdk_path))
        if asi.get_num_cameras() == 0:
            raise RuntimeError("no ASI camera detected")
        self._camera = asi.Camera(self._camera_id)
        props = self._camera.get_camera_property()
        self._width = props["MaxWidth"]
        self._height = props["MaxHeight"]
        self._is_color = bool(props["IsColorCam"])
        self._bayer_pattern = props.get("BayerPattern")
        self._camera.set_image_type(_BIT_DEPTH_TO_IMG_TYPE[self._bit_depth])

    def close(self) -> None:
        if self._camera is None:
            return
        try:
            self._camera.stop_video_capture()
        except asi.ZWO_Error:
            pass  # wasn't streaming — fine
        self._camera.close()
        self._camera = None

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        assert self._camera is not None
        # Changing ROI while video capture is running is NOT safe on real
        # hardware (confirmed on a real ASI290MC): the first 1-2 reads
        # after the change still succeed (stale/in-flight frames), then
        # capture_video_frame() times out permanently -- the sensor's
        # capture pipeline needs a stop/restart bracket around a format
        # change, exposure/gain don't need this (verified those stay live
        # while streaming). Mirrors ZWO's own documented usage pattern.
        #
        # The ASI SDK also requires width a multiple of 8 and height a
        # multiple of 2, and raises if not -- round rather than reject, so
        # this stays usable from a free-form mouse drag (confirmed on real
        # hardware: an unrounded width raises "ROI width must be multiple
        # of 8" from inside the SDK call).
        width = max(8, (width // 8) * 8)
        height = max(2, (height // 2) * 2)
        was_streaming = self._streaming
        if was_streaming:
            self._camera.stop_video_capture()
        try:
            self._camera.set_roi(start_x=x, start_y=y, width=width, height=height)
            self._width, self._height = width, height
        finally:
            # Must restart even if the SDK call above raised (e.g. ROI out
            # of sensor bounds) -- confirmed on real hardware: without
            # this try/finally, a rejected set_roi() left the stream
            # stopped while the worker's read loop kept calling
            # read_frame(), producing an infinite Timeout instead of a
            # clean error and a still-working camera.
            if was_streaming:
                self._camera.start_video_capture()

    def set_bit_depth(self, bit_depth: int) -> None:
        """Switches the live video path (read_frame, and by extension SER
        recording and FITS snapshots -- both just read whatever this
        produces) between RAW8 and RAW16. Same stop/restart bracket as
        set_roi -- a format change while capture_video_frame() is in
        flight is the class of bug fixed there."""
        assert self._camera is not None
        if bit_depth not in _BIT_DEPTH_TO_IMG_TYPE:
            raise ValueError(f"unsupported bit depth {bit_depth!r} (must be 8 or 16)")
        was_streaming = self._streaming
        if was_streaming:
            self._camera.stop_video_capture()
        self._camera.set_image_type(_BIT_DEPTH_TO_IMG_TYPE[bit_depth])
        self._bit_depth = bit_depth
        if was_streaming:
            self._camera.start_video_capture()

    def set_exposure_us(self, microseconds: int) -> None:
        assert self._camera is not None
        self._camera.set_control_value(asi.ASI_EXPOSURE, int(microseconds))

    def set_gain(self, gain: int) -> None:
        assert self._camera is not None
        self._camera.set_control_value(asi.ASI_GAIN, int(gain))

    def get_controls(self) -> dict:
        assert self._camera is not None
        return self._camera.get_controls()

    def start_streaming(self) -> None:
        assert self._camera is not None
        self._camera.start_video_capture()
        self._streaming = True

    def stop_streaming(self) -> None:
        assert self._camera is not None
        self._camera.stop_video_capture()
        self._streaming = False

    def read_frame(self, timeout_ms: int = 2000) -> np.ndarray:
        assert self._camera is not None
        return self._camera.capture_video_frame(timeout=timeout_ms)

    def get_dropped_frames(self) -> int:
        """Cumulative count of frames the sensor produced but the SDK's
        ring buffer overwrote before the host fetched them via
        read_frame() -- i.e. the host wasn't keeping up (USB bandwidth,
        exposure/fps mismatch, or a slow consumer downstream). Confirmed
        on a real ASI290MC: 0 while reading frames promptly, climbs (326
        over 2s at ~185fps) as soon as reads are neglected -- a real,
        useful signal, not just our own read_frame() timing/timeouts."""
        assert self._camera is not None
        return self._camera.get_dropped_frames()

    def bayer_pattern_ser_colour_id(self) -> int:
        if not self._is_color or self._bayer_pattern is None:
            return 0  # SER COLOURID_MONO
        return _BAYER_TO_SER_COLOUR_ID.get(self._bayer_pattern, 0)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_color(self) -> bool:
        return self._is_color

    @property
    def bit_depth(self) -> int:
        return self._bit_depth
