"""ASTAP-based plate solving.

Runs the ASTAP binary as a subprocess on a saved FITS/PNG snapshot,
parses the .ini result file ASTAP writes beside the input, and returns
the solved RA/DEC of the frame centre.

Disabled gracefully when ASTAP is not installed -- callers check
PlateSolver.available before using it, and the UI shows a clear message.

ASTAP CLI reference (confirmed against ASTAP v0.9.764):
  astap -f <file> -r <search_radius_deg> -fov <field_of_view_deg>
        -z 0              (no downsampling, our frames are already small)
        -o <output_dir>   (where .wcs/.ini are written)
Returns exit code 0 on success, non-zero on failure.
The .ini file beside the output contains:
  PLTSOLVD=T   (success flag)
  CRVAL1=<RA_deg>
  CRVAL2=<DEC_deg>
  CD1_1, CD1_2, CD2_1, CD2_2  (WCS matrix, pixel scale + rotation)
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _find_astap() -> str | None:
    """Returns the path to the ASTAP binary, or None if not found."""
    import shutil
    found = shutil.which("astap") or shutil.which("astap_cli")
    if found:
        return found
    # Common manual-install locations
    for candidate in [
        "/usr/local/bin/astap", "/opt/astap/astap",
        Path.home() / "bin" / "astap",          # user-local install (our default)
        Path.home() / "astap" / "astap",
    ]:
        p = Path(candidate)
        if p.exists():
            return str(p)
    return None


ASTAP_PATH: str | None = _find_astap()


@dataclass
class SolveResult:
    success: bool
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    field_rotation_deg: float = 0.0
    pixel_scale_arcsec: float = 0.0
    message: str = ""


class PlateSolver:
    """Thin wrapper around the ASTAP command-line solver.

    Instantiate once; call solve() from a background thread (it blocks
    for the solver duration, typically 1-10 s depending on hardware).
    The UI should use solve_async() instead, which runs it off the main
    thread and calls on_done(SolveResult) back on the caller's thread via
    Tk's after().
    """

    def __init__(self, astap_path: str | None = None, search_radius_deg: float = 30.0):
        self._astap = astap_path or ASTAP_PATH
        self._search_radius = search_radius_deg

    @property
    def available(self) -> bool:
        return self._astap is not None and Path(self._astap).exists()

    def solve(
        self,
        frame: np.ndarray,
        fov_deg: float = 1.0,
        hint_ra_deg: float | None = None,
        hint_dec_deg: float | None = None,
    ) -> SolveResult:
        """Synchronous solve -- call from a worker thread, not the UI thread."""
        if not self.available:
            return SolveResult(success=False, message=f"ASTAP not found (looked for: {self._astap or 'astap'})")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Write frame as FITS
            try:
                from astropy.io import fits
                fits_path = tmp / "frame.fits"
                fits.PrimaryHDU(data=frame.astype(np.float32)).writeto(fits_path)
            except Exception as exc:
                return SolveResult(success=False, message=f"Failed to write FITS: {exc}")

            cmd = [
                self._astap,
                "-f", str(fits_path),
                "-fov", str(fov_deg),
                "-r", str(self._search_radius),
                "-z", "0",
                "-o", str(tmp / "result"),
            ]
            if hint_ra_deg is not None and hint_dec_deg is not None:
                cmd += ["-ra", str(hint_ra_deg / 15.0), "-spd", str(hint_dec_deg + 90.0)]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                return SolveResult(success=False, message="ASTAP timed out (>60s)")
            except Exception as exc:
                return SolveResult(success=False, message=f"ASTAP failed to run: {exc}")

            # Parse the .ini result file ASTAP writes
            ini_path = tmp / "result.ini"
            if not ini_path.exists():
                # ASTAP sometimes writes beside the input
                ini_path = fits_path.with_suffix(".ini")
            if not ini_path.exists():
                return SolveResult(
                    success=False,
                    message=f"ASTAP exit {proc.returncode}, no result file. stderr: {proc.stderr[:200]}",
                )
            return _parse_astap_ini(ini_path)

    def solve_async(
        self,
        frame: np.ndarray,
        tk_widget,
        on_done,
        fov_deg: float = 1.0,
        hint_ra_deg: float | None = None,
        hint_dec_deg: float | None = None,
    ) -> None:
        """Non-blocking solve -- runs in a daemon thread, calls
        on_done(SolveResult) back via tk_widget.after(0, ...) so the
        callback safely touches Tk widgets from the main thread."""
        def _run() -> None:
            result = self.solve(frame, fov_deg, hint_ra_deg, hint_dec_deg)
            tk_widget.after(0, lambda: on_done(result))

        threading.Thread(target=_run, daemon=True).start()


def _parse_astap_ini(path: Path) -> SolveResult:
    kv: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            kv[k.strip()] = v.strip()

    if kv.get("PLTSOLVD", "F").upper() != "T":
        return SolveResult(success=False, message=f"PLTSOLVD=F in result: {kv.get('PLTSOLVD', '?')}")

    try:
        ra_deg = float(kv["CRVAL1"])
        dec_deg = float(kv["CRVAL2"])
    except (KeyError, ValueError) as exc:
        return SolveResult(success=False, message=f"Missing CRVAL1/CRVAL2: {exc}")

    # Pixel scale and rotation from WCS matrix CD1_1 etc.
    try:
        import math
        cd11 = float(kv.get("CD1_1", "0"))
        cd12 = float(kv.get("CD1_2", "0"))
        cd21 = float(kv.get("CD2_1", "0"))
        _cd22 = float(kv.get("CD2_2", "0"))  # noqa: F841
        pixel_scale_deg = math.sqrt(cd11 ** 2 + cd21 ** 2)
        rotation_deg = math.degrees(math.atan2(cd12, cd11))
    except Exception:
        pixel_scale_deg = 0.0
        rotation_deg = 0.0

    return SolveResult(
        success=True,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        field_rotation_deg=rotation_deg,
        pixel_scale_arcsec=pixel_scale_deg * 3600.0,
    )
