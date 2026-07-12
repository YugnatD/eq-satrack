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

PREVIEW_INTERVAL_S = 0.1  # ~10Hz -- plenty for a framing/focus preview, well under the 100-200fps capture rate
STATS_INTERVAL_S = 1.0


@dataclass
class CameraEvent:
    kind: str
    payload: dict = field(default_factory=dict)


def frame_to_pgm(frame) -> bytes:
    """Encode a 2D uint8 array as a binary PGM (P5) image — tk.PhotoImage
    reads this directly via data=..., no Pillow dependency needed. This is
    a raw-sensor grayscale preview (Bayer mosaic, not debayered), good
    enough for framing/focus, not a colour rendering."""
    height, width = frame.shape
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    return header + frame.tobytes()


def pgm_to_array(pgm: bytes) -> np.ndarray:
    """Inverse of frame_to_pgm -- decodes a binary PGM (P5) back into a 2D
    uint8 array, e.g. for camera/guiding.py's blob detection to run on a
    "preview_frame" event's payload without needing a second, un-throttled
    frame path from the camera."""
    header, _, body = pgm.partition(b"255\n")
    _, width_s, height_s = header.split()
    width, height = int(width_s), int(height_s)
    return np.frombuffer(body, dtype=np.uint8).reshape(height, width)


class CameraWorker:
    def __init__(self) -> None:
        self.events: "queue.Queue[CameraEvent]" = queue.Queue()
        self._commands: "queue.Queue[tuple[str, dict]]" = queue.Queue()
        self._camera: AsiCamera | MockAsiCamera | None = None
        self._streaming = threading.Event()
        self._ser_writer: SerWriter | None = None
        self._recording_path: Path | None = None
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- public, thread-safe API ------------------------------------------------

    def connect(
        self, kind: str, camera_id: int = 0, sdk_path: str | None = None, mock_seed: int | None = None,
        plate_scale_arcsec_per_px: float | None = None,
    ) -> None:
        self._commands.put(("connect", {
            "kind": kind, "camera_id": camera_id, "sdk_path": sdk_path, "mock_seed": mock_seed,
            "plate_scale_arcsec_per_px": plate_scale_arcsec_per_px,
        }))

    def disconnect(self) -> None:
        self._commands.put(("disconnect", {}))

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        self._commands.put(("set_roi", {"x": x, "y": y, "width": width, "height": height}))

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
        while not self._shutdown.is_set():
            try:
                name, payload = self._commands.get_nowait()
                handler = handlers.get(name)
                if handler is not None:
                    try:
                        handler(payload)
                    except Exception as exc:  # noqa: BLE001 - surface it, keep the worker alive
                        self._emit("log", message=f"[error] {name} failed: {exc}")
            except queue.Empty:
                pass

            if self._camera is None or not self._streaming.is_set():
                time.sleep(0.05)
                continue

            try:
                frame = self._camera.read_frame(timeout_ms=2000)
            except Exception as exc:  # noqa: BLE001 - a dropped/timed-out frame shouldn't kill the loop
                self._emit("log", message=f"[warn] read_frame failed: {exc}")
                continue

            frame_count_since_stats += 1
            now_utc = datetime.now(timezone.utc)
            if self._ser_writer is not None:
                self._ser_writer.add_frame(frame, timestamp=now_utc)

            now = time.monotonic()
            if now - last_preview >= PREVIEW_INTERVAL_S:
                last_preview = now
                self._emit("preview_frame", pgm=frame_to_pgm(frame), width=frame.shape[1], height=frame.shape[0])
            if now - last_stats >= STATS_INTERVAL_S:
                fps = frame_count_since_stats / (now - last_stats)
                self._emit("stats", fps=fps, recording=self._ser_writer is not None,
                            frames_recorded=self._ser_writer.frame_count if self._ser_writer else 0)
                last_stats = now
                frame_count_since_stats = 0

        self._handle_disconnect({})

    # -- command handlers -----------------------------------------------------

    def _handle_connect(self, payload: dict) -> None:
        kind = payload["kind"]
        if kind == "mock":
            mock_kwargs = {"seed": payload.get("mock_seed")}
            if payload.get("plate_scale_arcsec_per_px") is not None:
                mock_kwargs["plate_scale_arcsec_per_px"] = payload["plate_scale_arcsec_per_px"]
            camera = MockAsiCamera(**mock_kwargs)
        else:
            camera = AsiCamera(payload["camera_id"], payload.get("sdk_path"))
        try:
            camera.open()
        except Exception as exc:  # noqa: BLE001
            self._emit("connect_error", message=str(exc))
            return
        self._camera = camera
        self._camera.start_streaming()
        self._streaming.set()
        self._emit("connected", width=camera.width, height=camera.height, is_color=camera.is_color,
                    controls=camera.get_controls())

    def _handle_disconnect(self, payload: dict) -> None:
        self._streaming.clear()
        if self._ser_writer is not None:
            self._ser_writer.close()
            self._ser_writer = None
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        self._emit("disconnected")

    def _handle_set_roi(self, payload: dict) -> None:
        if self._camera is None:
            return
        self._camera.set_roi(payload["x"], payload["y"], payload["width"], payload["height"])

    def _handle_set_exposure_us(self, payload: dict) -> None:
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
        if self._ser_writer is not None:
            self._ser_writer.close()
        path = Path(payload["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ser_writer = SerWriter(
            path, width=self._camera.width, height=self._camera.height,
            colour_id=self._camera.bayer_pattern_ser_colour_id(), pixel_depth=self._camera.bit_depth,
            observer=payload.get("observer", ""), instrument=payload.get("instrument", ""),
            telescope=payload.get("telescope", ""),
        )
        self._recording_path = path
        self._emit("recording_started", path=str(path))

    def _handle_stop_recording(self, payload: dict) -> None:
        if self._ser_writer is None:
            return
        frame_count = self._ser_writer.frame_count
        self._ser_writer.close()
        self._ser_writer = None
        self._emit("recording_stopped", path=str(self._recording_path), frame_count=frame_count)
        self._recording_path = None

    def _handle_save_fits_snapshot(self, payload: dict) -> None:
        if self._camera is None:
            self._emit("log", message="[error] can't save snapshot: not connected")
            return
        frame = self._camera.read_frame(timeout_ms=5000)
        path = Path(payload["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_fits(path, frame, header_extra={"INSTRUME": "ASI290MC"})
        self._emit("fits_saved", path=str(path))
