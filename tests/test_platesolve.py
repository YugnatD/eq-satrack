import subprocess

import numpy as np
import pytest

from camera.platesolve import HINT_RADIUS_DEG, AstrometryNetSolver, _parse_astap_ini, _parse_astrometry_wcs


@pytest.fixture
def fake_solve_field(tmp_path):
    p = tmp_path / "solve-field"
    p.write_text("#!/bin/sh\n")
    p.chmod(0o755)
    return p


def test_default_timeout_is_short_enough_for_a_doomed_solve_not_to_stall_retries(fake_solve_field):
    # Regression: this used to default to 120s. A real "no solution"
    # attempt on real hardware measured 97.8s before failing (vs. 2-4s
    # for a real match) -- with POLAR_SOLVE_RETRY_ATTEMPTS/
    # SOLVE_RETRY_ATTEMPTS retrying up to 5 times, a single doomed point
    # could block for ~10 minutes. 30s keeps a comfortable ~10x margin
    # over a real solve's typical runtime while cutting failure-path cost.
    solver = AstrometryNetSolver(solve_field_path=str(fake_solve_field))
    assert solver._timeout_s == 30.0


def test_solve_passes_a_cpulimit_a_few_seconds_under_the_subprocess_timeout(fake_solve_field, monkeypatch):
    # Regression: solve-field used to have no --cpulimit, so a doomed
    # search only ever stopped via subprocess.run's own timeout (a hard
    # SIGKILL mid-search) -- passing --cpulimit lets solve-field exit on
    # its own a few seconds earlier, cleanly, with a real "no solution"
    # result instead of a bare TimeoutExpired.
    captured_cmd = {}

    def fake_run(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", fake_run)

    solver = AstrometryNetSolver(solve_field_path=str(fake_solve_field), timeout_s=30.0)
    frame = np.zeros((10, 10), dtype=np.uint8)
    result = solver.solve(frame, fov_deg=1.0)

    assert not result.success
    cmd = captured_cmd["cmd"]
    assert "--cpulimit" in cmd
    cpulimit_value = cmd[cmd.index("--cpulimit") + 1]
    assert cpulimit_value == "25"


def test_solve_uses_a_hint_radius_independent_of_fov_not_derived_from_it(fake_solve_field, monkeypatch):
    # Regression, found on real hardware: the --radius passed alongside
    # --ra/--dec used to be max(fov_deg, 1.0) -- the CAMERA's own imaging
    # field, unrelated to how wrong the position HINT itself can be. A
    # PAA point failed all 5 retries on genuinely different, clean,
    # star-rich frames because the mount's own hint (Dec~90 after park +
    # RA-only rotation) was 2.49deg off the true position -- just outside
    # the old ~1.83deg (fov_deg) radius, so solve-field searched the
    # wrong patch of sky every single attempt. Re-solving the exact same
    # frame offline with a wider radius found the real match in ~1s. A
    # small fov_deg here must not shrink the radius below HINT_RADIUS_DEG.
    captured_cmd = {}

    def fake_run(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", fake_run)

    solver = AstrometryNetSolver(solve_field_path=str(fake_solve_field))
    frame = np.zeros((10, 10), dtype=np.uint8)
    solver.solve(frame, fov_deg=1.833, hint_ra_deg=54.8125, hint_dec_deg=90.0)

    cmd = captured_cmd["cmd"]
    radius_value = float(cmd[cmd.index("--radius") + 1])
    assert radius_value == HINT_RADIUS_DEG
    assert radius_value > 1.833


def test_parse_astrometry_wcs_sets_flip_parity_from_the_real_cd_matrix(tmp_path):
    # Regression, found on real hardware: SolveResult used to have no
    # parity information at all -- am5/polar_alignment.py's
    # project_radec_to_pixel assumed one fixed parity (negative
    # determinant), but our actual finder camera's real solves have a
    # POSITIVE determinant, confirmed against a real .wcs file from a
    # live PAA run. Positive-determinant fixture here.
    from astropy.io import fits

    header = fits.Header()
    header["CRVAL1"] = 173.828804
    header["CRVAL2"] = 87.510850
    header["CD1_1"] = -0.000458878557259
    header["CD1_2"] = -8.70769270881e-05
    header["CD2_1"] = 8.71527322194e-05
    header["CD2_2"] = -0.000458933357235
    path = tmp_path / "result.wcs"
    fits.PrimaryHDU(header=header).writeto(path)

    result = _parse_astrometry_wcs(path)
    assert result.success
    assert result.flip_parity is True


def test_parse_astrometry_wcs_negative_determinant_is_not_flipped(tmp_path):
    from astropy.io import fits

    header = fits.Header()
    header["CRVAL1"] = 30.0
    header["CRVAL2"] = 60.0
    header["CD1_1"] = 0.0005
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = -0.0005
    path = tmp_path / "result.wcs"
    fits.PrimaryHDU(header=header).writeto(path)

    result = _parse_astrometry_wcs(path)
    assert result.success
    assert result.flip_parity is False


def test_parse_astap_ini_sets_flip_parity_from_the_real_cd_matrix(tmp_path):
    path = tmp_path / "result.ini"
    path.write_text(
        "PLTSOLVD=T\n"
        "CRVAL1=173.828804\n"
        "CRVAL2=87.510850\n"
        "CD1_1=-0.000458878557259\n"
        "CD1_2=-8.70769270881e-05\n"
        "CD2_1=8.71527322194e-05\n"
        "CD2_2=-0.000458933357235\n"
    )
    result = _parse_astap_ini(path)
    assert result.success
    assert result.flip_parity is True
