import math

import numpy as np

from camera.finder import FinderCalibration, FinderState


def test_main_fov_corners_px_returns_none_when_not_calibrated():
    c = FinderCalibration()
    assert c.main_fov_corners_px((2160, 3840), 1936, 1096) is None


def test_main_fov_corners_px_centred_no_rotation():
    c = FinderCalibration()
    c.calibrated = True
    c.offset_row = 0.0
    c.offset_col = 0.0
    c.plate_scale_ratio = 2.0  # finder arcsec/px is 2x the main's -- main FOV shrinks by 2x in finder px
    c.rotation_rad = 0.0

    corners = c.main_fov_corners_px((2160, 3840), main_width_px=1000, main_height_px=800)

    rows = [r for r, _ in corners]
    cols = [cc for _, cc in corners]
    assert min(rows) == 1080.0 - 200.0  # 800/2/2
    assert max(rows) == 1080.0 + 200.0
    assert min(cols) == 1920.0 - 250.0  # 1000/2/2
    assert max(cols) == 1920.0 + 250.0


def test_main_fov_corners_px_applies_offset():
    c = FinderCalibration()
    c.calibrated = True
    c.offset_row = 10.0
    c.offset_col = -20.0
    c.plate_scale_ratio = 1.0
    c.rotation_rad = 0.0

    corners = c.main_fov_corners_px((200, 200), main_width_px=40, main_height_px=40)
    centre_row = sum(r for r, _ in corners) / 4.0
    centre_col = sum(cc for _, cc in corners) / 4.0
    assert centre_row == 100.0 + 10.0
    assert centre_col == 100.0 - 20.0


def test_main_fov_corners_px_rotation_rotates_rectangle():
    c = FinderCalibration()
    c.calibrated = True
    c.offset_row = 0.0
    c.offset_col = 0.0
    c.plate_scale_ratio = 1.0
    c.rotation_rad = math.pi / 2.0  # 90 degrees -- width/height swap in effect

    corners = c.main_fov_corners_px((200, 200), main_width_px=40, main_height_px=20)
    rows = [r for r, _ in corners]
    cols = [cc for _, cc in corners]
    # After a 90deg rotation, the half-extents along row/col swap: the
    # rectangle's row-span should now reflect the original half-width (20),
    # not the original half-height (10).
    assert max(rows) - min(rows) == 40.0
    assert max(cols) - min(cols) == 20.0


def test_update_main_frame_populates_last_main_frame():
    state = FinderState()
    assert state.last_main_frame is None
    frame = np.zeros((10, 10), dtype=np.uint8)
    state.update_main_frame(frame)
    assert state.last_main_frame is frame


def test_get_correction_arcsec_defaults_to_the_real_configured_finder_scale():
    state = FinderState()
    state.finder_plate_scale_arcsec = 1.72
    state.calibration.calibrated = True
    state.last_frame = np.zeros((100, 100), dtype=np.uint8)
    state.blob_found = True
    state.last_blob_row = 60.0  # 10px below center
    state.last_blob_col = 50.0  # centered horizontally

    default_call = state.get_correction_arcsec()
    explicit_call = state.get_correction_arcsec(finder_plate_scale_arcsec=1.72)
    assert default_call == explicit_call

    wrong_scale_call = state.get_correction_arcsec(finder_plate_scale_arcsec=1.0)
    assert default_call != wrong_scale_call


def test_main_fov_corners_px_applies_roi_offset():
    c = FinderCalibration()
    c.calibrated = True
    c.offset_row = 0.0
    c.offset_col = 0.0
    c.plate_scale_ratio = 1.0
    c.rotation_rad = 0.0

    # A 40x40 ROI whose centre sits 30px right / 10px down from the main
    # sensor's own optical centre -- e.g. a smaller ROI dragged off-centre.
    corners = c.main_fov_corners_px(
        (200, 200), main_width_px=40, main_height_px=40,
        main_roi_offset_row_px=10.0, main_roi_offset_col_px=30.0,
    )
    centre_row = sum(r for r, _ in corners) / 4.0
    centre_col = sum(cc for _, cc in corners) / 4.0
    assert centre_row == 100.0 + 10.0
    assert centre_col == 100.0 + 30.0
    # Size unaffected by the offset -- still a 40x40 ROI at ratio 1.0.
    rows = [r for r, _ in corners]
    cols = [cc for _, cc in corners]
    assert max(rows) - min(rows) == 40.0
    assert max(cols) - min(cols) == 40.0


def test_main_fov_corners_px_roi_offset_scaled_by_plate_ratio():
    c = FinderCalibration()
    c.calibrated = True
    c.offset_row = 0.0
    c.offset_col = 0.0
    c.plate_scale_ratio = 2.0  # finder arcsec/px is 2x the main's
    c.rotation_rad = 0.0

    corners = c.main_fov_corners_px(
        (200, 200), main_width_px=40, main_height_px=40,
        main_roi_offset_row_px=0.0, main_roi_offset_col_px=100.0,
    )
    centre_col = sum(cc for _, cc in corners) / 4.0
    # 100 main px of offset / plate_scale_ratio 2.0 = 50 finder px.
    assert centre_col == 100.0 + 50.0


def test_state_main_fov_corners_px_passes_through_roi_offset():
    state = FinderState()
    state.calibration.calibrated = True
    state.calibration.plate_scale_ratio = 1.0
    state.last_frame = np.zeros((200, 200), dtype=np.uint8)
    state.main_sensor_width = 40
    state.main_sensor_height = 40
    state.main_roi_offset_row = 10.0
    state.main_roi_offset_col = 30.0

    corners = state.main_fov_corners_px()
    centre_row = sum(r for r, _ in corners) / 4.0
    centre_col = sum(cc for _, cc in corners) / 4.0
    assert centre_row == 100.0 + 10.0
    assert centre_col == 100.0 + 30.0


def test_reset_blob_clears_stale_detection():
    state = FinderState()
    state.blob_found = True
    state.last_blob_row = 42.0
    state.last_blob_col = 17.0

    state.reset_blob()

    assert state.blob_found is False
    assert state.last_blob_row is None
    assert state.last_blob_col is None


def test_reset_blob_prevents_stale_correction_after_disconnect():
    state = FinderState()
    state.finder_plate_scale_arcsec = 1.72
    state.calibration.calibrated = True
    state.last_frame = np.zeros((100, 100), dtype=np.uint8)
    state.blob_found = True
    state.last_blob_row = 60.0
    state.last_blob_col = 50.0
    assert state.get_correction_arcsec() is not None

    state.reset_blob()

    assert state.get_correction_arcsec() is None


def test_calibrate_from_frames_sets_rotation_from_rotation_deg():
    c = FinderCalibration()
    frame = np.random.default_rng(1).random((50, 50))
    c.calibrate_from_frames(frame, frame, rotation_deg=12.5)
    assert c.rotation_rad == math.radians(12.5)


def test_calibrate_from_frames_defaults_rotation_to_zero():
    c = FinderCalibration()
    c.rotation_rad = 99.0  # a stale value from a previous calibration
    frame = np.random.default_rng(1).random((50, 50))
    c.calibrate_from_frames(frame, frame)
    assert c.rotation_rad == 0.0
