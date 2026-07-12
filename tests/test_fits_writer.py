import numpy as np
from astropy.io import fits

from camera.fits_writer import write_fits


def test_write_fits_round_trip(tmp_path):
    path = tmp_path / "snapshot.fits"
    frame = np.arange(64, dtype=np.uint8).reshape(8, 8)
    write_fits(path, frame, header_extra={"INSTRUME": "ASI290MC", "GAIN": 300, "EXPTIME": 0.001})

    with fits.open(path) as hdul:
        assert np.array_equal(hdul[0].data, frame)
        assert hdul[0].header["INSTRUME"] == "ASI290MC"
        assert hdul[0].header["GAIN"] == 300
        assert "DATE-OBS" in hdul[0].header
