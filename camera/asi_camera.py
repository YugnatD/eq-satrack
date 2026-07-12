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


class AsiCamera:
    """A real ZWO ASI camera, opened via the vendor SDK (zwoasi). RAW8 by
    default (no 16-bit/colour-debayer conversion) — the right format for
    high framerate planetary/ISS capture where bandwidth is the constraint."""

    def __init__(self, camera_id: int = 0, sdk_path: str | None = None):
        self._camera_id = camera_id
        self._sdk_path = sdk_path
        self._camera: asi.Camera | None = None
        self._width = 0
        self._height = 0
        self._is_color = False
        self._bayer_pattern: int | None = None
        self._bit_depth = 8

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
        self._camera.set_image_type(asi.ASI_IMG_RAW8)

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
        self._camera.set_roi(start_x=x, start_y=y, width=width, height=height)
        self._width, self._height = width, height

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

    def stop_streaming(self) -> None:
        assert self._camera is not None
        self._camera.stop_video_capture()

    def read_frame(self, timeout_ms: int = 2000) -> np.ndarray:
        assert self._camera is not None
        return self._camera.capture_video_frame(timeout=timeout_ms)

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
