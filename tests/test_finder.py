import math

import numpy as np
import pytest

from camera.finder import FinderCalibration, FinderState
from camera.guiding import GuidingCalibration


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


def test_finder_px_to_main_px_zero_at_the_main_cameras_own_centre():
    # Regression: this used to ADD offset_row/offset_col instead of
    # SUBTRACTING them -- inconsistent with this same class's own
    # main_fov_corners_px (centre_row = fc_row + offset_row, verified by
    # the synthetic-ground-truth tests below), which means the main
    # camera's own optical centre, expressed in finder pixels, is
    # (fc_row + offset_row, fc_col + offset_col). A blob sitting exactly
    # there is BY DEFINITION at zero offset from the main camera's centre.
    c = FinderCalibration()
    c.calibrated = True
    c.offset_row = 10.0
    c.offset_col = -20.0
    c.plate_scale_ratio = 2.0
    c.rotation_rad = 0.0

    finder_shape = (200, 300)
    fc_row, fc_col = 100.0, 150.0
    dx_px, dy_px = c.finder_px_to_main_px(fc_row + 10.0, fc_col - 20.0, finder_shape)
    assert dx_px == pytest.approx(0.0)
    assert dy_px == pytest.approx(0.0)


def test_finder_px_to_main_px_scales_by_plate_ratio():
    c = FinderCalibration()
    c.calibrated = True
    c.offset_row = 0.0
    c.offset_col = 0.0
    c.plate_scale_ratio = 2.0  # finder arcsec/px is 2x the main's
    c.rotation_rad = 0.0

    # 5 finder px east of the main centre -> 10 main px (main px are finer).
    dx_px, dy_px = c.finder_px_to_main_px(100.0, 105.0, (200, 200))
    assert dx_px == pytest.approx(10.0)
    assert dy_px == pytest.approx(0.0)


def test_finder_px_to_main_px_applies_rotation():
    c = FinderCalibration()
    c.calibrated = True
    c.offset_row = 0.0
    c.offset_col = 0.0
    c.plate_scale_ratio = 1.0
    c.rotation_rad = math.pi / 2.0

    # Pure row offset (10 finder px "down" from main centre) should come
    # out as a pure column offset after a 90deg roll.
    dx_px, dy_px = c.finder_px_to_main_px(110.0, 100.0, (200, 200))
    assert dx_px == pytest.approx(10.0)
    assert dy_px == pytest.approx(0.0)


def test_get_correction_arcsec_requires_the_main_cameras_own_sky_calibration():
    # Regression: a finder-to-main geometric calibration alone (FinderCame
    # raPanel's "Calibrate fields") only knows the finder's roll/scale
    # relative to the MAIN camera's own pixel axes -- not how those axes
    # relate to true sky directions. Only the main camera's own nudge-
    # verified GuidingCalibration (CalibrationPanel's "Calibrate camera-
    # to-sky mapping") establishes that, so get_correction_arcsec must
    # refuse to answer without it rather than silently assuming the main
    # camera's axes happen to already be sky-track-aligned.
    state = FinderState()
    state.calibration.calibrated = True
    state.last_frame = np.zeros((100, 100), dtype=np.uint8)
    state.blob_found = True
    state.last_blob_row = 60.0
    state.last_blob_col = 50.0

    assert state.main_calibration is None
    assert state.get_correction_arcsec() is None

    state.set_main_calibration(GuidingCalibration(
        px_per_ra_arcsec_x=1.0, px_per_ra_arcsec_y=0.0, px_per_dec_arcsec_x=0.0, px_per_dec_arcsec_y=1.0,
    ))
    assert state.get_correction_arcsec() is not None


def test_get_correction_arcsec_chains_through_the_main_calibration():
    state = FinderState()
    state.calibration.calibrated = True
    state.calibration.offset_row = 0.0
    state.calibration.offset_col = 0.0
    state.calibration.plate_scale_ratio = 1.0
    state.calibration.rotation_rad = 0.0
    state.last_frame = np.zeros((100, 100), dtype=np.uint8)
    state.blob_found = True
    state.last_blob_row = 50.0  # exactly at finder centre == main centre here
    state.last_blob_col = 60.0  # 10px east of main centre

    calib = GuidingCalibration(
        px_per_ra_arcsec_x=2.0, px_per_ra_arcsec_y=0.0, px_per_dec_arcsec_x=0.0, px_per_dec_arcsec_y=2.0,
    )
    state.set_main_calibration(calib)

    result = state.get_correction_arcsec()
    expected = calib.pixel_to_sky(10.0, 0.0)
    assert result == pytest.approx(expected)


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
    state.calibration.calibrated = True
    state.last_frame = np.zeros((100, 100), dtype=np.uint8)
    state.blob_found = True
    state.last_blob_row = 60.0
    state.last_blob_col = 50.0
    state.set_main_calibration(GuidingCalibration(
        px_per_ra_arcsec_x=1.0, px_per_ra_arcsec_y=0.0, px_per_dec_arcsec_x=0.0, px_per_dec_arcsec_y=1.0,
    ))
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


def test_calibrate_from_frames_recovers_a_known_offset_across_different_plate_scales():
    # Regression test for a real bug caught on actual hardware (real Vega
    # captures, real ASI290MC/ASI678MM optics): the previous implementation
    # resized the FINDER frame down to match the MAIN frame's own PIXEL
    # COUNT, then ran a whole-frame FFT phase correlation -- only valid if
    # both cameras see roughly the same total angular field, which is false
    # by construction here (the finder's field is much wider than the main
    # camera's). The resulting FOV rectangle didn't contain Vega even
    # though the main camera's own simultaneous capture did.
    #
    # Ground truth built independently of calibrate_from_frames's own
    # logic: a "sky" noise field at the main camera's fine angular
    # resolution, covering the finder's whole (wider) field. main_frame is
    # a raw crop of that sky at a KNOWN pixel location; finder_frame is the
    # whole sky resampled down to the finder's coarser resolution. The
    # true centre of main's content, in finder pixels, is computable
    # directly from the known crop location and the plate scale ratio --
    # calibrate_from_frames must recover that same centre without being
    # given it.
    rng = np.random.default_rng(3)
    finder_h, finder_w = 400, 600
    finder_scale = 2.0
    main_h, main_w = 100, 150
    main_scale = 0.5
    ratio = finder_scale / main_scale  # 4.0 -- finder sees 4x more sky per pixel

    sky_h, sky_w = round(finder_h * ratio), round(finder_w * ratio)
    sky = np.clip(rng.normal(size=(sky_h, sky_w)), 0, None)

    known_top, known_left = 700, 1200
    main_frame = sky[known_top:known_top + main_h, known_left:known_left + main_w].copy()
    from skimage.transform import resize
    finder_frame = resize(sky, (finder_h, finder_w), anti_aliasing=True)

    true_centre_row = (known_top + main_h / 2) / ratio
    true_centre_col = (known_left + main_w / 2) / ratio

    c = FinderCalibration()
    c.calibrate_from_frames(main_frame, finder_frame, main_plate_scale_arcsec=main_scale, finder_plate_scale_arcsec=finder_scale)
    corners = c.main_fov_corners_px((finder_h, finder_w), main_w, main_h)
    rows = [r for r, _ in corners]
    cols = [_c for _, _c in corners]
    computed_centre_row = (min(rows) + max(rows)) / 2
    computed_centre_col = (min(cols) + max(cols)) / 2

    assert computed_centre_row == pytest.approx(true_centre_row, abs=1.0)
    assert computed_centre_col == pytest.approx(true_centre_col, abs=1.0)
    # The rectangle must actually be SMALLER than the finder frame (main's
    # field is narrower) -- the old bug's pixel-count-based resize made it
    # roughly half the finder frame's size regardless of the true ratio.
    assert (max(rows) - min(rows)) < finder_h / 2
    assert (max(cols) - min(cols)) < finder_w / 2


def test_calibrate_from_frames_the_recovered_target_lands_inside_the_main_fov_rectangle():
    # A second, independent framing of the same regression: a target that
    # is genuinely within the main camera's field (placed at main's own
    # centre) must land inside the calibrated FOV rectangle drawn on the
    # finder frame -- this is the exact real-world symptom that surfaced
    # the bug (a real captured Vega, confirmed present in the main
    # camera's own simultaneous frame, fell outside the rectangle).
    rng = np.random.default_rng(7)
    finder_h, finder_w = 500, 700
    finder_scale = 1.7189  # this project's real SVBony 60mm F4 + ASI678MM
    main_h, main_w = 1096, 1936
    main_scale = 0.5982  # this project's real 1000mm F/4 + ASI290MC
    ratio = finder_scale / main_scale

    sky_h, sky_w = round(finder_h * ratio), round(finder_w * ratio)
    sky = np.clip(rng.normal(size=(sky_h, sky_w)), 0, None)
    top, left = sky_h // 3, sky_w // 3  # anywhere but dead centre, to actually exercise the offset
    main_frame = sky[top:top + main_h, left:left + main_w].copy()
    from skimage.transform import resize
    finder_frame = resize(sky, (finder_h, finder_w), anti_aliasing=True)

    target_row_finder = (top + main_h / 2) / ratio
    target_col_finder = (left + main_w / 2) / ratio

    c = FinderCalibration()
    c.calibrate_from_frames(main_frame, finder_frame, main_plate_scale_arcsec=main_scale, finder_plate_scale_arcsec=finder_scale)
    corners = c.main_fov_corners_px((finder_h, finder_w), main_w, main_h)
    rows = [r for r, _ in corners]
    cols = [_c for _, _c in corners]

    assert min(rows) <= target_row_finder <= max(rows)
    assert min(cols) <= target_col_finder <= max(cols)
