import numpy as np
import pytest

from camera.guiding import GuidingCalibration, calibrate_from_nudges, detect_brightest_blob, measure_frame_shift


def _synthetic_frame(cx: float, cy: float, width: int = 200, height: int = 150, peak: float = 200.0, sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    background = np.random.default_rng(1).normal(20.0, 3.0, size=(height, width))
    blob = peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
    return np.clip(background + blob, 0, 255).astype(np.uint8)


def test_detect_brightest_blob_finds_known_centroid():
    frame = _synthetic_frame(cx=120.0, cy=60.0)
    blob = detect_brightest_blob(frame)
    assert blob.found is True
    assert blob.centroid_x == pytest.approx(120.0, abs=1.0)
    assert blob.centroid_y == pytest.approx(60.0, abs=1.0)


def test_detect_brightest_blob_not_found_on_pure_noise():
    rng = np.random.default_rng(2)
    frame = np.clip(rng.normal(20.0, 3.0, size=(150, 200)), 0, 255).astype(np.uint8)
    blob = detect_brightest_blob(frame)
    assert blob.found is False


def test_detect_brightest_blob_rejects_single_hot_pixel():
    frame = _synthetic_frame(cx=50.0, cy=50.0, peak=0.0)  # no real blob, just background
    frame = frame.copy()
    frame[10, 10] = 255  # one stray hot pixel
    blob = detect_brightest_blob(frame, min_pixels=5)
    assert blob.found is False


def test_calibrate_from_nudges_pure_x_and_y_axes():
    # RA nudge moves the blob purely in +x; DEC nudge purely in +y
    calib = calibrate_from_nudges(
        d_ra_arcsec=10.0, ra_nudge_dx_px=50.0, ra_nudge_dy_px=0.0,
        d_dec_arcsec=10.0, dec_nudge_dx_px=0.0, dec_nudge_dy_px=30.0,
    )
    dx, dy = calib.sky_to_pixel(d_ra_arcsec=2.0, d_dec_arcsec=1.0)
    assert dx == pytest.approx(10.0)  # 2 * (50/10)
    assert dy == pytest.approx(3.0)  # 1 * (30/10)


def test_calibrate_from_nudges_roundtrip_pixel_to_sky():
    calib = calibrate_from_nudges(
        d_ra_arcsec=10.0, ra_nudge_dx_px=40.0, ra_nudge_dy_px=15.0,
        d_dec_arcsec=8.0, dec_nudge_dx_px=-12.0, dec_nudge_dy_px=25.0,
    )
    d_ra, d_dec = 3.5, -2.1
    dx, dy = calib.sky_to_pixel(d_ra, d_dec)
    d_ra_rt, d_dec_rt = calib.pixel_to_sky(dx, dy)
    assert d_ra_rt == pytest.approx(d_ra)
    assert d_dec_rt == pytest.approx(d_dec)


def test_calibrate_from_nudges_raises_on_zero_sky_motion():
    with pytest.raises(ValueError):
        calibrate_from_nudges(d_ra_arcsec=0.0, ra_nudge_dx_px=10.0, ra_nudge_dy_px=0.0,
                               d_dec_arcsec=10.0, dec_nudge_dx_px=0.0, dec_nudge_dy_px=10.0)


def test_pixel_to_sky_raises_on_degenerate_matrix():
    # both nudges produced the same pixel direction -- matrix isn't invertible
    calib = GuidingCalibration(px_per_ra_arcsec_x=1.0, px_per_ra_arcsec_y=1.0, px_per_dec_arcsec_x=2.0, px_per_dec_arcsec_y=2.0)
    with pytest.raises(ValueError):
        calib.pixel_to_sky(5.0, 5.0)


def _synthetic_star_field(width: int = 240, height: int = 200, seed: int = 3) -> np.ndarray:
    # A handful of point sources (not just one blob) plus noise -- phase
    # correlation needs real image texture to lock onto, unlike
    # detect_brightest_blob's single-source case above.
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    field = rng.normal(20.0, 3.0, size=(height, width))
    for cx, cy, peak in [(40, 30, 180.0), (150, 60, 220.0), (90, 140, 150.0), (200, 170, 190.0), (60, 100, 160.0)]:
        field += peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.5**2)))
    return np.clip(field, 0, 255).astype(np.uint8)


def test_measure_frame_shift_recovers_a_known_pixel_shift():
    reference = _synthetic_star_field()
    # np.roll shifts array CONTENT by (+3 rows, -5 cols) -- the star field
    # itself moved that much, which is what measure_frame_shift's own
    # (delta_col, delta_row) return should report (in this project's
    # column=x/row=y convention, not skimage's row/col array order).
    live = np.roll(reference, shift=(3, -5), axis=(0, 1))

    delta_col, delta_row = measure_frame_shift(reference, live, downsample=1)

    assert delta_col == pytest.approx(-5.0, abs=0.5)
    assert delta_row == pytest.approx(3.0, abs=0.5)


def test_measure_frame_shift_survives_downsampling():
    reference = _synthetic_star_field()
    live = np.roll(reference, shift=(-8, 12), axis=(0, 1))

    result = measure_frame_shift(reference, live, downsample=4)

    assert result is not None
    delta_col, delta_row = result
    assert delta_col == pytest.approx(12.0, abs=2.0)
    assert delta_row == pytest.approx(-8.0, abs=2.0)


def test_measure_frame_shift_returns_none_for_unrelated_frames():
    # A frame with no star field in common at all -- e.g. the live view
    # has drifted the star field out of frame, or a cloud rolled through
    # -- must not silently report a meaningless shift.
    reference = _synthetic_star_field(seed=3)
    rng = np.random.default_rng(99)
    unrelated = np.clip(rng.normal(20.0, 3.0, size=reference.shape), 0, 255).astype(np.uint8)

    assert measure_frame_shift(reference, unrelated, downsample=1) is None


def test_measure_frame_shift_handles_mismatched_frame_sizes():
    reference = _synthetic_star_field(width=240, height=200)
    live = np.roll(reference, shift=(2, -1), axis=(0, 1))[:180, :220]  # smaller live frame, as if the camera ROI changed

    result = measure_frame_shift(reference, live, downsample=1)

    assert result is not None
    delta_col, delta_row = result
    assert delta_col == pytest.approx(-1.0, abs=0.5)
    assert delta_row == pytest.approx(2.0, abs=0.5)
