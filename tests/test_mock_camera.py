import numpy as np
import pytest

from camera.mock_camera import DEFAULT_ARCSEC_PER_PIXEL, MockAsiCamera

ISS_PEAK_SANITY_THRESHOLD = 150.0  # ISS peak is 200*gain_scale; background stars are much dimmer


def test_read_frame_shape_matches_roi():
    cam = MockAsiCamera(seed=1)
    cam.open()
    cam.set_roi(0, 0, 320, 240)
    cam.start_streaming()
    frame = cam.read_frame()
    assert frame.shape == (240, 320)
    assert frame.dtype == np.uint8
    cam.close()


def test_read_frame_contains_a_bright_blob_somewhere():
    cam = MockAsiCamera(seed=2)
    cam.open()
    cam.set_roi(0, 0, 200, 150)
    cam.set_exposure_us(500)
    cam.start_streaming()
    frame = cam.read_frame()
    # background is centered ~20 with small sigma; a real blob pixel should
    # stand out well above that
    assert frame.max() > 60
    cam.close()


def test_sky_context_places_iss_at_offset_from_boresight():
    cam = MockAsiCamera(seed=4, plate_scale_arcsec_per_px=DEFAULT_ARCSEC_PER_PIXEL)
    cam.open()
    cam.set_roi(0, 0, 200, 150)
    cam.start_streaming()

    boresight_ra, boresight_dec = 45.0, 10.0
    dx_arcsec = 20 * DEFAULT_ARCSEC_PER_PIXEL  # ~20px right
    dy_arcsec = -10 * DEFAULT_ARCSEC_PER_PIXEL  # ~10px down (DEC+ is image-up, so negative dy moves down)
    target_ra = boresight_ra + (dx_arcsec / 3600.0) / np.cos(np.radians(boresight_dec))
    target_dec = boresight_dec + dy_arcsec / 3600.0
    cam.set_sky_context(boresight_ra, boresight_dec, target_ra, target_dec)
    frame = cam.read_frame().astype(float)

    peak_y, peak_x = np.unravel_index(np.argmax(frame), frame.shape)
    assert peak_x == pytest.approx(100 + 20, abs=3)
    assert peak_y == pytest.approx(75 + 10, abs=3)
    cam.close()


def test_sky_context_target_equals_boresight_puts_iss_at_roi_center():
    cam = MockAsiCamera(seed=5)
    cam.open()
    cam.set_roi(0, 0, 200, 150)
    cam.start_streaming()
    cam.set_sky_context(45.0, 10.0, 45.0, 10.0)
    frame = cam.read_frame().astype(float)

    peak_y, peak_x = np.unravel_index(np.argmax(frame), frame.shape)
    assert peak_x == pytest.approx(100, abs=3)
    assert peak_y == pytest.approx(75, abs=3)
    cam.close()


def test_sky_context_far_off_frame_shows_no_iss_blob():
    cam = MockAsiCamera(seed=6)
    cam.open()
    cam.set_roi(0, 0, 200, 150)
    cam.start_streaming()
    # target 10000 pixels' worth of arcsec away in RA -- way outside the ROI
    far_ra = 45.0 + (10_000 * DEFAULT_ARCSEC_PER_PIXEL / 3600.0)
    cam.set_sky_context(45.0, 10.0, far_ra, 10.0)
    frame = cam.read_frame().astype(float)

    # background + dim background stars only -- no ISS-brightness peak
    assert frame.max() < ISS_PEAK_SANITY_THRESHOLD


def test_sky_context_pans_background_stars_with_boresight():
    # Same target-minus-boresight offset (ISS stays centered), but two very
    # different boresight pointings -- if a star is visible near one
    # boresight, the *pixel it appears at* must differ once we point
    # elsewhere (the field pans), even though the ISS itself stays centered.
    cam = MockAsiCamera(seed=7)
    cam.open()
    cam.set_roi(0, 0, 400, 300)
    cam.start_streaming()

    vega_ra, vega_dec = 279.23410832, 38.78299311  # HIP 91262

    # boresight #1: pointed straight at Vega
    cam.set_sky_context(vega_ra, vega_dec, vega_ra, vega_dec + 5.0)
    frame1 = cam.read_frame().astype(float)

    # boresight #2: pointed 90 deg away -- Vega is nowhere in frame anymore
    cam.set_sky_context((vega_ra + 90.0) % 360.0, vega_dec, (vega_ra + 90.0) % 360.0, vega_dec + 5.0)
    frame2 = cam.read_frame().astype(float)

    assert not np.allclose(frame1, frame2, atol=1.0)
    cam.close()


def test_pointing_at_vega_shows_a_real_bright_star_at_center():
    cam = MockAsiCamera(seed=8, plate_scale_arcsec_per_px=2.0)
    cam.open()
    cam.set_roi(0, 0, 300, 200)
    cam.start_streaming()

    vega_ra, vega_dec = 279.23410832, 38.78299311  # HIP 91262, Vmag 0.03
    # target far outside the frame -- isolate the star field from the ISS blob
    cam.set_sky_context(vega_ra, vega_dec, (vega_ra + 90.0) % 360.0, vega_dec)
    frame = cam.read_frame().astype(float)

    peak_y, peak_x = np.unravel_index(np.argmax(frame), frame.shape)
    assert peak_x == pytest.approx(150, abs=3)
    assert peak_y == pytest.approx(100, abs=3)
    assert frame.max() > 150  # Vega is very bright (mag 0.03) -- should saturate near the sensor ceiling
    cam.close()


def test_fainter_star_renders_dimmer_than_a_brighter_one():
    from camera.mock_camera import _load_star_catalog

    star_ra, star_dec, star_mag = _load_star_catalog()
    bright_idx = int(np.argmin(star_mag))
    # a star several magnitudes fainter than the catalog's brightest, but
    # still within the naked-eye cutoff, for a meaningful (not degenerate) comparison
    faint_idx = int(np.argmin(np.abs(star_mag - (star_mag[bright_idx] + 4.0))))
    bright_ra, bright_dec = float(star_ra[bright_idx]), float(star_dec[bright_idx])
    faint_ra, faint_dec = float(star_ra[faint_idx]), float(star_dec[faint_idx])

    def peak_at(ra: float, dec: float) -> float:
        cam = MockAsiCamera(seed=9, plate_scale_arcsec_per_px=1.0)
        cam.open()
        cam.set_roi(0, 0, 60, 60)
        cam.start_streaming()
        # target far outside the frame -- isolate the star field from the
        # (much brighter) ISS blob, which would otherwise render dead
        # center and saturate regardless of which star is under test.
        cam.set_sky_context(ra, dec, (ra + 90.0) % 360.0, dec)
        frame = cam.read_frame().astype(float)
        cam.close()
        return float(frame.max())

    assert peak_at(bright_ra, bright_dec) > peak_at(faint_ra, faint_dec)


def test_gain_increases_noise_and_signal():
    low = MockAsiCamera(seed=3)
    low.open()
    low.set_roi(0, 0, 100, 100)
    low.set_gain(0)
    low.start_streaming()
    low_frame = low.read_frame().astype(float)

    high = MockAsiCamera(seed=3)
    high.open()
    high.set_roi(0, 0, 100, 100)
    high.set_gain(570)
    high.start_streaming()
    high_frame = high.read_frame().astype(float)

    assert high_frame.std() > low_frame.std()
    low.close()
    high.close()


def test_longer_exposure_reveals_a_fainter_star():
    # Previously exposure_us only paced frame timing and had zero effect on
    # brightness -- gain was the only way to reveal fainter stars. A longer
    # exposure should integrate more signal, same as a real sensor.
    from camera.mock_camera import _load_star_catalog

    star_ra, star_dec, star_mag = _load_star_catalog()
    bright_idx = int(np.argmin(star_mag))
    faint_idx = int(np.argmin(np.abs(star_mag - (star_mag[bright_idx] + 4.0))))
    faint_ra, faint_dec = float(star_ra[faint_idx]), float(star_dec[faint_idx])

    def peak_at(exposure_us: int) -> float:
        cam = MockAsiCamera(seed=10, plate_scale_arcsec_per_px=1.0)
        cam.open()
        cam.set_roi(0, 0, 60, 60)
        cam.set_exposure_us(exposure_us)
        cam.start_streaming()
        cam.set_sky_context(faint_ra, faint_dec, (faint_ra + 90.0) % 360.0, faint_dec)
        frame = cam.read_frame().astype(float)
        cam.close()
        return float(frame.max())

    assert peak_at(20_000) > peak_at(1_000)
