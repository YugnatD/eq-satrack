"""Finder-scope camera support: a wide-field second camera that helps
acquire the ISS when the main camera's FOV is too narrow to guarantee
a hit at pass start.

Two independent features, both optional:

1. ISS blob detection in the finder preview (reuses detect_brightest_blob
   from camera/guiding.py -- the ISS stands out strongly against the sky
   even at wide field) -- gives an "ISS is HERE in the finder" overlay.

2. Field correlation: cross-correlates finder and main frames on their
   shared star field (stars are stationary for the ~6 min pass, unlike
   the ISS) to find the pixel offset between the two camera centres.
   Uses FFT phase cross-correlation (skimage.registration) -- no
   catalogue, no plate-solving, sub-pixel accuracy in ~ms.
   Result: a translation (dx, dy) in finder pixels. After calibrating
   the finder's plate scale vs. the main camera's (one arcsec per pixel
   each), this translates directly to an along/cross-track correction.

Usage:
    FinderCalibration.calibrate(main_frame, finder_frame)
        → stores offset + scale; call once before the pass.
    FinderCalibration.correction_arcsec(main_frame, finder_frame)
        → (along_arcsec, cross_arcsec) correction for LiveOffsets.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

# Shared by both finder exposure sliders (am5/gui/finder_window.py's
# FinderWindow and am5/gui/panels.py's FinderCameraPanel) -- a wide-field
# finder is used to ACQUIRE a target quickly, not for long deep-sky
# integration, so there's no real use case for exposures anywhere near the
# slider's old 10^9us (1000s) ceiling; capped here instead at 5s, still far
# beyond anything a finder acquisition exposure would need.
MAX_FINDER_EXPOSURE_US = 5_000_000


def downsample_for_display(
    gray_frame: np.ndarray, canvas_w: int, canvas_h: int,
) -> tuple[int, int, float, np.ndarray]:
    """Shrinks a (potentially large) 2D frame to roughly fit a canvas, for
    a live preview -- via integer-stride slicing, NOT skimage.transform.
    resize. resize()'s interpolation costs ~240ms on a 3840x2592 frame
    (measured on this machine); slicing costs ~2µs. At the finder's real
    sensor size and a ~10Hz preview rate, resize() on the Tk main thread
    (see FinderWindow/FinderCameraPanel.handle_camera_event, called from
    App._pump_events) visibly froze the whole app -- confirmed by
    connecting the mock finder camera and watching it hang. A live
    preview doesn't need resize()'s smoothing; a slightly blocky downscale
    is imperceptible at preview size and costs nothing.

    Returns (display_width, display_height, scale_factor, display_array)
    -- scale_factor converts full-frame pixel coordinates (e.g. a blob
    position) to display coordinates."""
    fh, fw = gray_frame.shape[:2]
    scale = min(canvas_w / fw, canvas_h / fh)
    stride = max(1, int(round(1.0 / scale))) if scale > 0 else 1
    small = gray_frame[::stride, ::stride]
    sh, sw = small.shape[:2]
    actual_scale = sw / fw if fw > 0 else 1.0
    display = small if small.dtype == np.uint8 else small.astype(np.uint8)
    return sw, sh, actual_scale, display


def _to_gray_float(frame: np.ndarray) -> np.ndarray:
    """8-bit or 16-bit, colour or mono → float64 2D array, normalised 0-1."""
    if frame.ndim == 3:
        frame = frame.mean(axis=2)
    f = frame.astype(np.float64)
    mx = f.max()
    return f / mx if mx > 0 else f


@dataclass
class FinderCalibration:
    """Stores the geometric relationship between finder and main camera.
    All pixel coordinates are in FINDER-camera space.

    plate_scale_ratio: arcsec/px_finder ÷ arcsec/px_main -- if the
    finder has a wider FOV and larger plate scale, this is > 1.  Set by
    calibrate() from the observed shift vs. known nudge (same logic as
    GuidingCalibration), or set manually from known specs.

    rotation_rad: angle of the finder's x-axis relative to the main
    camera's x-axis (usually ~0 if both cameras share the same mount).
    """
    offset_row: float = 0.0       # finder centre row relative to main centre, finder px
    offset_col: float = 0.0       # finder centre col relative to main centre, finder px
    plate_scale_ratio: float = 1.0
    rotation_rad: float = 0.0
    calibrated: bool = False

    def calibrate_from_frames(
        self, main_frame: np.ndarray, finder_frame: np.ndarray,
        main_plate_scale_arcsec: float = 1.0, finder_plate_scale_arcsec: float = 1.0,
        rotation_deg: float = 0.0,
    ) -> None:
        """Locates the main camera's (narrow) field within the finder's
        (wide) field by template matching. Both frames must be taken at
        the SAME sky position (point both cameras roughly at the same
        region before calibrating).

        Regression fix -- this used to resize the FINDER frame down to
        match the MAIN frame's own PIXEL COUNT, then run a whole-frame
        FFT phase correlation between the two. That's only meaningful if
        both cameras see roughly the same total angular field of view --
        false by construction here (the finder's field is typically many
        times wider than the main camera's, e.g. ~1.8deg vs ~0.3deg for
        this project's actual rig, confirmed against real hardware specs)
        -- so a whole-frame correlation was comparing two images at
        completely different angular scales and could not recover a
        physically meaningful offset. Confirmed broken two independent
        ways on real Vega captures: the resulting FOV rectangle didn't
        contain Vega even though the main camera's own simultaneous
        capture did, and the computed offset didn't match a direct,
        independent measurement of Vega's pixel position in each frame
        converted through the real, user-provided focal lengths/pixel
        sizes.

        Fixed by actually using the plate scale ratio: shrink the MAIN
        frame down to the finder's own (coarser) angular resolution --
        now a main pixel and a finder pixel represent the same real
        angular size -- then search for that shrunk main image as a
        template WITHIN the (full, unmodified) finder frame via
        normalized cross-correlation (skimage.feature.match_template).
        This is the correct shape for the problem (find where a small
        detail sits within a wider view), unlike a global translation
        search between two full frames of very different content extent.
        Verified against a synthetic pair built with a known plate-scale
        ratio and a known embed location: the recovered offset matched
        the true location to within rounding.

        rotation_deg: the finder's mechanical roll relative to the main
        camera, in degrees -- NOT measured here. Template matching alone
        doesn't recover rotation, and this project's own star fields are
        too sparse (a handful of point sources, not a textured scene) for
        a joint rotation+translation search to be reliable -- a wrong
        blind estimate would silently corrupt corrections worse than the
        honest default. If the finder is mounted with a visible roll
        offset from the main tube, measure it by hand (e.g. compare a
        known star's position angle in both cameras) and pass it here --
        the caller (FinderCameraPanel) exposes this as a plain entry field
        rather than guessing, so the assumption is visible, not silent.
        """
        from skimage.feature import match_template
        from skimage.transform import resize

        ma = _to_gray_float(main_frame)
        fa = _to_gray_float(finder_frame)
        ratio = finder_plate_scale_arcsec / max(main_plate_scale_arcsec, 1e-9)
        # Shrink main to the finder's own angular pixel scale -- clamped
        # to fa's own shape (min 1px) so a misconfigured/inverted scale
        # (ratio <= 1, main claiming a WIDER field than the finder) still
        # produces a valid, if degraded, template rather than crashing
        # inside match_template (which requires template <= image).
        small_h = max(1, min(fa.shape[0], round(ma.shape[0] / ratio)))
        small_w = max(1, min(fa.shape[1], round(ma.shape[1] / ratio)))
        ma_small = resize(ma, (small_h, small_w), anti_aliasing=True)
        response = match_template(fa, ma_small, pad_input=True, mode="constant")
        peak_row, peak_col = np.unravel_index(np.argmax(response), response.shape)
        self.offset_row = float(peak_row) - fa.shape[0] / 2.0
        self.offset_col = float(peak_col) - fa.shape[1] / 2.0
        self.plate_scale_ratio = ratio
        self.rotation_rad = math.radians(rotation_deg)
        self.calibrated = True

    def finder_px_to_correction_arcsec(
        self, finder_blob_row: float, finder_blob_col: float,
        finder_frame_shape: tuple[int, int],
        finder_plate_scale_arcsec: float = 1.0,
    ) -> tuple[float, float]:
        """Given a detected ISS blob position in the finder frame, returns
        (delta_row_arcsec, delta_col_arcsec) -- how far the main camera
        needs to move to centre the ISS.  Positive = move DOWN/RIGHT."""
        if not self.calibrated:
            return 0.0, 0.0
        # Finder centre
        fc_row = finder_frame_shape[0] / 2.0
        fc_col = finder_frame_shape[1] / 2.0
        # Blob offset from finder centre, in finder pixels
        dr = finder_blob_row - fc_row + self.offset_row
        dc = finder_blob_col - fc_col + self.offset_col
        # Convert to arcseconds using finder plate scale
        dr_arcsec = dr * finder_plate_scale_arcsec
        dc_arcsec = dc * finder_plate_scale_arcsec
        # Apply rotation to align with main camera axes
        cos_r, sin_r = math.cos(self.rotation_rad), math.sin(self.rotation_rad)
        along = dr_arcsec * cos_r - dc_arcsec * sin_r
        cross = dr_arcsec * sin_r + dc_arcsec * cos_r
        return along, cross

    def main_fov_corners_px(
        self, finder_frame_shape: tuple[int, int],
        main_width_px: int, main_height_px: int,
        main_roi_offset_row_px: float = 0.0, main_roi_offset_col_px: float = 0.0,
    ) -> list[tuple[float, float]] | None:
        """Corners (row, col), in FINDER pixel space, of the rectangle the
        main camera actually captures -- for drawing "this is what the
        acquisition camera sees" on top of the finder's wider preview.
        None if not calibrated (the offset/rotation/scale ratio would be
        meaningless). Uses plate_scale_ratio (finder arcsec/px over main
        arcsec/px, set by calibrate_from_frames) to convert the main
        sensor's own pixel size into finder pixels: a main-camera pixel
        covers more sky per pixel than a finder pixel when the finder has
        the shorter focal length, so its footprint SHRINKS in finder space
        by that same ratio.

        main_width_px/main_height_px are the main camera's CURRENT capture
        extent, not necessarily its full sensor -- pass the active ROI's
        size (FinderState.main_sensor_width/height already carries this,
        see TransitPanel._apply_roi) so the rectangle shrinks to match a
        smaller ROI instead of always claiming the full sensor's field.
        main_roi_offset_row_px/col_px: the ROI centre's own offset from
        the full sensor's optical centre, in MAIN camera pixels (0 if the
        ROI is centred, e.g. the default full-sensor case) -- also
        converted through plate_scale_ratio and rotated, same as the
        rectangle's own extents, since it's a real angular displacement
        too, not just a cosmetic nudge.
        """
        if not self.calibrated:
            return None
        ratio = max(self.plate_scale_ratio, 1e-9)
        cos_r, sin_r = math.cos(self.rotation_rad), math.sin(self.rotation_rad)
        offset_row_finder = main_roi_offset_row_px / ratio
        offset_col_finder = main_roi_offset_col_px / ratio
        rotated_offset_row = offset_row_finder * cos_r - offset_col_finder * sin_r
        rotated_offset_col = offset_row_finder * sin_r + offset_col_finder * cos_r
        centre_row = finder_frame_shape[0] / 2.0 + self.offset_row + rotated_offset_row
        centre_col = finder_frame_shape[1] / 2.0 + self.offset_col + rotated_offset_col
        half_h = main_height_px / ratio / 2.0
        half_w = main_width_px / ratio / 2.0
        corners = []
        for dr, dc in ((-half_h, -half_w), (-half_h, half_w), (half_h, half_w), (half_h, -half_w)):
            rr = dr * cos_r - dc * sin_r
            rc = dr * sin_r + dc * cos_r
            corners.append((centre_row + rr, centre_col + rc))
        return corners


@dataclass
class FinderState:
    """All mutable state for the finder camera, safe to share across threads.
    Owned by FinderCameraPanel; read by TransitPanel for corrections.

    update_frame() runs on the Tk main thread (called from App._pump_events,
    see FinderCameraPanel/FinderWindow.handle_camera_event) -- unlike the
    small main-camera preview (max ~640x480), the finder's real sensor is
    3840x2592, and detect_brightest_blob's own np.mgrid/centroid math on a
    frame that size costs ~70ms measured on this machine. At the ~10Hz
    preview_frame rate, running that on the UI thread every single frame
    visibly freezes the whole app (confirmed: connecting the mock finder
    camera froze the window). Two mitigations, both cheap and enough to
    stay smooth without touching the shared/proven camera worker itself:
    downsample BEFORE detecting (the blob's few-pixel footprint survives a
    4x reduction fine, and the cost drops ~16x) and skip most frames
    entirely (blob position only needs to update a few times a second for
    a human to track it, not at the full preview rate)."""
    calibration: FinderCalibration = field(default_factory=FinderCalibration)
    last_frame: np.ndarray | None = None
    last_blob_row: float | None = None
    last_blob_col: float | None = None
    blob_found: bool = False
    # Set by TransitPanel._apply_roi -- shared here (rather than passed
    # separately to each preview widget) since FinderState is the one
    # instance both FinderCameraPanel and FinderWindow already hold, and
    # both need it to draw the main camera's FOV rectangle on top of the
    # finder's wider view. These reflect the main camera's CURRENT capture
    # extent -- the active ROI, not necessarily the full sensor (set to
    # the full sensor at connect, since _apply_roi(0, 0, sensor_w,
    # sensor_h) is what TransitPanel's own "connected" handler calls) --
    # so a smaller ROI shrinks the rectangle to match instead of the
    # rectangle silently overstating the real captured field.
    main_sensor_width: int | None = None
    main_sensor_height: int | None = None
    # The active ROI's own centre offset from the full sensor's optical
    # centre, in MAIN camera pixels -- 0 for a centred/full-sensor ROI.
    # See FinderCalibration.main_fov_corners_px's docstring for why this
    # needs the same scale+rotation treatment as the rectangle's extents.
    main_roi_offset_row: float = 0.0
    main_roi_offset_col: float = 0.0
    # Latest main-camera frame, for calibrate_from_frames -- set by App from
    # the main camera's own "preview_frame" event (see update_main_frame),
    # same rationale as main_sensor_width/height above: this is the one
    # shared instance both the calibration UI and the correction reader
    # already hold, so there's a single place for the main camera's data to
    # land instead of each widget needing its own private plumbing.
    last_main_frame: np.ndarray | None = None
    # Real, currently-configured plate scales (arcsec/px) for each camera --
    # set by ConnectionPanel at connect time from whatever optics are
    # actually configured (mock focal/pixel fields, or the Exposure calc
    # tab's optical train for a real main camera). Used by calibrate_from_
    # frames and get_correction_arcsec below instead of a separately typed,
    # easily-stale duplicate value.
    main_plate_scale_arcsec: float = 1.0
    finder_plate_scale_arcsec: float = 1.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _frame_counter: int = field(default=0, repr=False, compare=False)
    # Every Nth frame gets blob-detected -- at a ~10Hz preview rate this is
    # ~2.5Hz, plenty responsive for a human tracking a slow-moving dot.
    _detect_every_n: int = field(default=4, repr=False, compare=False)
    # Detection runs on a downsampled copy -- see class docstring.
    _detect_downsample: int = field(default=4, repr=False, compare=False)

    def update_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self.last_frame = frame
            self._frame_counter += 1
            should_detect = self._frame_counter % self._detect_every_n == 0
        if not should_detect:
            return
        from camera.guiding import detect_brightest_blob
        small = frame[:: self._detect_downsample, :: self._detect_downsample]
        blob = detect_brightest_blob(small)
        with self._lock:
            self.blob_found = blob.found
            if blob.found:
                self.last_blob_row = blob.centroid_y * self._detect_downsample
                self.last_blob_col = blob.centroid_x * self._detect_downsample

    def update_main_frame(self, frame: np.ndarray) -> None:
        """Called from App on every main-camera preview_frame event (see
        App._pump_events) -- keeps last_main_frame current so 'Calibrate
        fields' can correlate against a REAL main-camera frame instead of
        silently falling back to correlating the finder against itself."""
        with self._lock:
            self.last_main_frame = frame

    def main_fov_corners_px(self) -> list[tuple[float, float]] | None:
        """Corners (row, col) in finder-pixel space of the main camera's
        FOV rectangle, or None if not calibrated yet or the main camera's
        sensor size isn't known yet (see main_sensor_width/height)."""
        with self._lock:
            if self.last_frame is None or self.main_sensor_width is None or self.main_sensor_height is None:
                return None
            frame_shape = self.last_frame.shape[:2]
            offset_row, offset_col = self.main_roi_offset_row, self.main_roi_offset_col
        return self.calibration.main_fov_corners_px(
            frame_shape, self.main_sensor_width, self.main_sensor_height, offset_row, offset_col,
        )

    def reset_blob(self) -> None:
        """Called when the finder camera disconnects (see App._pump_events)
        -- clears the last detected ISS blob so a stale, no-longer-real
        position can't keep silently driving finder corrections after the
        camera stops streaming (get_correction_arcsec would otherwise keep
        returning the last value forever, since nothing else invalidates
        it)."""
        with self._lock:
            self.blob_found = False
            self.last_blob_row = None
            self.last_blob_col = None

    def get_correction_arcsec(
        self, finder_plate_scale_arcsec: float | None = None,
    ) -> tuple[float, float] | None:
        """Returns (along_arcsec, cross_arcsec) if a blob is detected AND
        the calibration has been done, else None. Uses this instance's own
        finder_plate_scale_arcsec (the real, currently-configured value --
        see the field's docstring) unless the caller explicitly overrides
        it."""
        scale = finder_plate_scale_arcsec if finder_plate_scale_arcsec is not None else self.finder_plate_scale_arcsec
        with self._lock:
            if not self.blob_found or self.last_blob_row is None or self.last_frame is None:
                return None
            if not self.calibration.calibrated:
                return None
            return self.calibration.finder_px_to_correction_arcsec(
                self.last_blob_row, self.last_blob_col,
                self.last_frame.shape[:2], scale,
            )
