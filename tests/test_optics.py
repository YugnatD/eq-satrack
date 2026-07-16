import numpy as np
import pytest

from am5.optics import (
    OpticalTrain,
    estimate_iss_magnitude,
    estimate_signal_electrons,
    max_exposure_s,
    render_iss_photo,
    suggest_gain,
)


def test_optical_train_rejects_zero_focal_length():
    # Regression: a fat-fingered "0" focal length parses fine as a float
    # (no ValueError from the GUI's own field parsing), so callers that
    # only guarded that parsing went on to divide by an effective focal
    # length of zero inside plate_scale_arcsec_per_px -- an uncaught
    # ZeroDivisionError, confirmed to leave ConnectionPanel's connect
    # button stuck disabled at "Connecting..." with no error and no way to
    # retry short of restarting the app. Raising ValueError here instead
    # (same exception type every call site already catches around its own
    # float(...) parsing) turns the crash into an ordinary "invalid input".
    with pytest.raises(ValueError):
        OpticalTrain(aperture_mm=200, focal_length_mm=0.0, barlow_multiplier=1.0, pixel_size_um=2.9)


def test_optical_train_rejects_zero_barlow():
    with pytest.raises(ValueError):
        OpticalTrain(aperture_mm=200, focal_length_mm=1000.0, barlow_multiplier=0.0, pixel_size_um=2.9)


def test_optical_train_rejects_zero_pixel_size():
    with pytest.raises(ValueError):
        OpticalTrain(aperture_mm=200, focal_length_mm=1000.0, barlow_multiplier=1.0, pixel_size_um=0.0)


def test_optical_train_rejects_negative_focal_length():
    with pytest.raises(ValueError):
        OpticalTrain(aperture_mm=200, focal_length_mm=-1000.0, barlow_multiplier=1.0, pixel_size_um=2.9)


def test_suggest_gain_matches_real_verified_capture():
    # 200mm aperture, 1000mm FL, no barlow, 1ms exposure, mag -3.3 -> the
    # operator empirically used gain 220 (0.1dB units) for this real ISS
    # capture. MAG0_PHOTON_FLUX_PER_CM2_S was recalibrated against exactly
    # this data point -- this test locks that calibration in.
    train = OpticalTrain(aperture_mm=200, focal_length_mm=1000, barlow_multiplier=1.0, pixel_size_um=2.9)
    signal = estimate_signal_electrons(train, magnitude=-3.3, exposure_s=0.001)
    gain = suggest_gain(signal)
    assert gain == pytest.approx(220.0, abs=1.0)


def test_plate_scale_smaller_pixels_or_longer_focal_gives_finer_scale():
    train = OpticalTrain(aperture_mm=200, focal_length_mm=1000, barlow_multiplier=1.0, pixel_size_um=2.9)
    scale = train.plate_scale_arcsec_per_px
    assert scale == pytest.approx(206264.8 * 0.0029 / 1000.0)

    with_barlow = OpticalTrain(aperture_mm=200, focal_length_mm=1000, barlow_multiplier=2.0, pixel_size_um=2.9)
    assert with_barlow.plate_scale_arcsec_per_px == pytest.approx(scale / 2.0)


def test_max_exposure_shorter_for_faster_target():
    train = OpticalTrain(aperture_mm=200, focal_length_mm=1000, barlow_multiplier=1.0, pixel_size_um=2.9)
    slow = max_exposure_s(train, angular_speed_deg_s=0.1)
    fast = max_exposure_s(train, angular_speed_deg_s=1.0)
    assert fast < slow
    assert slow > 0


def test_max_exposure_infinite_for_stationary_target():
    train = OpticalTrain(aperture_mm=200, focal_length_mm=1000, barlow_multiplier=1.0, pixel_size_um=2.9)
    assert max_exposure_s(train, angular_speed_deg_s=0.0) == float("inf")


def test_iss_magnitude_dimmer_when_farther():
    close = estimate_iss_magnitude(distance_km=500.0)
    far = estimate_iss_magnitude(distance_km=2000.0)
    assert far > close  # higher (less negative/more positive) magnitude = dimmer


def test_signal_electrons_increase_with_aperture_and_exposure():
    small = OpticalTrain(aperture_mm=100, focal_length_mm=1000, barlow_multiplier=1.0, pixel_size_um=2.9)
    big = OpticalTrain(aperture_mm=300, focal_length_mm=1000, barlow_multiplier=1.0, pixel_size_um=2.9)
    s_small = estimate_signal_electrons(small, magnitude=-2.0, exposure_s=0.001)
    s_big = estimate_signal_electrons(big, magnitude=-2.0, exposure_s=0.001)
    assert s_big > s_small > 0

    s_longer = estimate_signal_electrons(small, magnitude=-2.0, exposure_s=0.01)
    assert s_longer > s_small


def test_suggest_gain_zero_when_already_bright_enough():
    assert suggest_gain(signal_electrons=1_000_000.0) == 0.0


def test_suggest_gain_positive_and_clamped_when_signal_is_faint():
    gain = suggest_gain(signal_electrons=1.0)
    assert 0.0 < gain <= 570.0


def test_render_iss_photo_more_camera_pixels_with_stronger_barlow():
    no_barlow = OpticalTrain(aperture_mm=200, focal_length_mm=1000, barlow_multiplier=1.0, pixel_size_um=2.9)
    with_barlow = OpticalTrain(aperture_mm=200, focal_length_mm=1000, barlow_multiplier=3.0, pixel_size_um=2.9)
    p1 = render_iss_photo(no_barlow, distance_km=500.0)
    p3 = render_iss_photo(with_barlow, distance_km=500.0)
    assert p3.camera_px_span > p1.camera_px_span  # finer plate scale -> more camera pixels across the same object
    assert p3.angular_size_arcsec == pytest.approx(p1.angular_size_arcsec)  # same real object, same sky angle


def test_render_iss_photo_display_size_is_constant_regardless_of_setup():
    # the whole point: switching optical trains must not resize the
    # rendered image -- only how blocky/detailed it looks inside it.
    coarse = OpticalTrain(aperture_mm=60, focal_length_mm=300, barlow_multiplier=1.0, pixel_size_um=2.9)
    fine = OpticalTrain(aperture_mm=200, focal_length_mm=1000, barlow_multiplier=3.0, pixel_size_um=2.9)
    p_coarse = render_iss_photo(coarse, distance_km=2000.0)
    p_fine = render_iss_photo(fine, distance_km=400.0)
    assert p_coarse.image.shape == p_fine.image.shape == (360, 360)


def test_render_iss_photo_image_is_grayscale_uint8_and_nonempty():
    train = OpticalTrain(aperture_mm=200, focal_length_mm=1000, barlow_multiplier=1.0, pixel_size_um=2.9)
    preview = render_iss_photo(train, distance_km=500.0)
    assert preview.image.dtype == np.uint8
    assert preview.image.ndim == 2
    assert preview.image.max() > 0  # the resampled photo actually has content
    assert preview.truncated is False


def test_render_iss_photo_flags_truncation_for_huge_objects():
    train = OpticalTrain(aperture_mm=200, focal_length_mm=10000, barlow_multiplier=5.0, pixel_size_um=2.9)
    preview = render_iss_photo(train, distance_km=400.0, max_native_px=20)
    assert preview.truncated is True
    assert preview.image.shape == (360, 360)  # still fits the fixed frame, just heavily downscaled to get there


def test_render_iss_photo_undersampled_still_shows_visible_blocks():
    # a coarse plate scale (small aperture, long distance) resamples the
    # photo down to only a handful of camera pixels -- must still render
    # as visibly blocky content within the fixed frame, not a near-empty canvas.
    train = OpticalTrain(aperture_mm=60, focal_length_mm=300, barlow_multiplier=1.0, pixel_size_um=2.9)
    preview = render_iss_photo(train, distance_km=2000.0)
    assert preview.camera_px_span < 80
    assert preview.image.shape == (360, 360)
    assert (preview.image > 10).sum() > 100  # meaningfully more than a few stray pixels lit
    assert max(preview.image.shape) >= 80
