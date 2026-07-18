"""Plate solving, via either of two interchangeable backends: ASTAP
(PlateSolver) or astrometry.net's solve-field (AstrometryNetSolver). Both
expose the same solve()/solve_async()/available interface and return the
same SolveResult, so callers (e.g. AlignmentPanel's polar-alignment tab)
can let the operator pick whichever is installed/working for them without
caring which one is actually running underneath.

Runs the solver binary as a subprocess on a saved FITS snapshot and parses
whatever result file it writes. Disabled gracefully when a given backend
isn't installed -- callers check <solver>.available before using it, and
the UI shows a clear message.

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

astrometry.net CLI reference (confirmed against astrometry.net 0.89):
  solve-field --no-plots --overwrite -D <dir> -o <basename>
              --scale-units degwidth --scale-low <lo> --scale-high <hi>
              [--ra <deg> --dec <deg> --radius <deg>]   (optional hint)
              <file>
  Writes <basename>.solved (present = success) and <basename>.wcs (a FITS
  header with CRVAL1/CRVAL2 + CD matrix, same fields as ASTAP's .ini)
  next to the input. Needs index files installed separately (see
  docs -- /usr/share/astrometry, indices matched to the expected FOV
  range) -- with none installed it just reports no solution, same as
  ASTAP with no star database.

  Known packaging bug (Ubuntu 22.04's astrometry.net 0.89, numpy>=2.0):
  solve-field shells out to internal helpers (astrometry.util.removelines,
  .uniformize) that reference the numpy<2.0 alias np.string_, which numpy
  2.0 removed -- see _ensure_numpy2_shim() below for how this is worked
  around without touching the system package.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# See AstrometryNetSolver.solve's own comment at its use site (the
# --radius flag) for the real-hardware failure this fixes -- a generous,
# fixed bound on how wrong the mount's own position hint can be, not the
# camera's field of view.
HINT_RADIUS_DEG = 10.0


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
    # True when the solved CD matrix has a POSITIVE determinant -- the
    # opposite parity from what am5/polar_alignment.py's
    # project_radec_to_pixel originally assumed (see its own docstring on
    # flip_parity for the real-hardware bug this fixes: confirmed our
    # actual finder camera solves with det(CD) > 0). Must be threaded
    # through to project_radec_to_pixel's own flip_parity parameter for
    # the polar-alignment overlay to point the correct on-screen
    # direction -- computed from the real solved matrix here, never
    # guessed at the call site.
    flip_parity: bool = False


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
        _solve_async(self.solve, frame, tk_widget, on_done, fov_deg, hint_ra_deg, hint_dec_deg)


def _astrometry_python_package_root() -> Path | None:
    """Locate the system astrometry.net Python package (the one
    solve-field's helper scripts, e.g. -m astrometry.util.uniformize,
    resolve via the default sys.path) -- not the CLI binaries."""
    for base in ("/usr/lib/python3/dist-packages", "/usr/lib/python3.10/dist-packages", "/usr/local/lib/python3.10/dist-packages"):
        p = Path(base) / "astrometry"
        if (p / "util" / "fits.py").exists():
            return p
    return None


def _ensure_numpy2_shim() -> Path | None:
    """Builds (once, cached) a local shadow copy of the astrometry.net
    Python package in a user cache dir, identical to the real system
    package except util/fits.py has np.string_ -> np.bytes_ (numpy
    renamed it in 2.0; same underlying type, upstream already fixed this
    in newer astrometry.net releases -- Ubuntu 22.04 ships 0.89, which
    still has the old name and crashes solve-field's internal
    removelines/uniformize helpers under numpy>=2.0).

    Every other file is a symlink to the real system file, so this stays
    a few KB (the compiled .so extensions and 50+ .py files aren't
    copied) and always reflects the real install except for the one
    patched line. Returns the shim's root dir (to prepend to
    PYTHONPATH), or None if the system package couldn't be found.

    Deliberately does NOT edit /usr/lib/python3/dist-packages/astrometry
    in place -- that file is shared by every program on the machine that
    imports it, not just this one, so the fix is scoped to solve-field
    subprocesses launched from here instead."""
    system_root = _astrometry_python_package_root()
    if system_root is None:
        return None

    shim_root = Path.home() / ".cache" / "satellite_tracking" / "astrometry_numpy2_shim"
    shim_pkg = shim_root / "astrometry"
    marker = shim_pkg / ".shim_source"
    if marker.exists() and marker.read_text() == str(system_root):
        return shim_root

    if shim_pkg.exists():
        shutil.rmtree(shim_pkg)
    shim_pkg.mkdir(parents=True)

    for item in system_root.rglob("*"):
        rel = item.relative_to(system_root)
        if "__pycache__" in rel.parts:
            continue
        dest = shim_pkg / rel
        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif rel == Path("util/fits.py"):
            dest.write_text(item.read_text().replace("np.string_", "np.bytes_"))
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(item)

    marker.write_text(str(system_root))
    return shim_root


class AstrometryNetSolver:
    """Same interface as PlateSolver, backed by astrometry.net's
    solve-field instead of ASTAP -- see the module docstring for the CLI
    reference. Instantiate once; call solve() from a background thread
    (typically several seconds, can be slower than ASTAP depending on
    which index files are installed), or solve_async() from the UI."""

    def __init__(self, solve_field_path: str | None = None, timeout_s: float = 30.0):
        # Was 120.0 -- confirmed live on real hardware that a real solve
        # here is either fast (2-4s, real match found) or doomed (index
        # files exhausted, no shortcut to detect this early): a genuine
        # "no solution" attempt measured 97.8s before this fix, so close
        # to the old 120s cap that 5 retries (POLAR_SOLVE_RETRY_ATTEMPTS/
        # SOLVE_RETRY_ATTEMPTS) could block for ~10 minutes on one point
        # that was never going to solve. 30s gives a real solve's typical
        # few-second runtime a ~10x margin while cutting the failure-path
        # cost roughly 3-4x, so retries reach a fresher frame sooner.
        import shutil
        self._solve_field = solve_field_path or shutil.which("solve-field")
        self._timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return self._solve_field is not None and Path(self._solve_field).exists()

    def solve(
        self,
        frame: np.ndarray,
        fov_deg: float = 1.0,
        hint_ra_deg: float | None = None,
        hint_dec_deg: float | None = None,
    ) -> SolveResult:
        """Synchronous solve -- call from a worker thread, not the UI thread."""
        if not self.available:
            return SolveResult(success=False, message=f"solve-field not found (looked for: {self._solve_field or 'solve-field'})")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            try:
                from astropy.io import fits
                fits_path = tmp / "frame.fits"
                fits.PrimaryHDU(data=frame.astype(np.float32)).writeto(fits_path)
            except Exception as exc:
                return SolveResult(success=False, message=f"Failed to write FITS: {exc}")

            cmd = [
                self._solve_field,
                "--no-plots", "--overwrite", "--no-verify",
                "-9",  # --no-remove-lines: its line-density heuristic takes log(mean)
                       # of per-row/column background bins, which is -inf/NaN (RuntimeWarning,
                       # then a spurious no-solution) on a mostly-black, sparse-star frame --
                       # meant for real-camera scanning artifacts (bad columns, satellite
                       # trails) in wide deep-sky images, not relevant to our narrow
                       # point-source fields, so always safe to skip.
                "-D", str(tmp), "-o", "result",
                "--scale-units", "degwidth",
                "--scale-low", str(fov_deg * 0.7),
                "--scale-high", str(fov_deg * 1.3),
                # Let solve-field itself give up and exit cleanly a few
                # seconds before our own subprocess timeout below --
                # avoids a hard SIGKILL mid-search (which on some
                # astrometry.net builds can leave a helper process
                # orphaned) and gets a real "no solution" message instead
                # of a bare TimeoutExpired.
                "--cpulimit", str(max(5, round(self._timeout_s - 5))),
            ]
            if hint_ra_deg is not None and hint_dec_deg is not None:
                # Regression, found on real hardware: this used to be
                # max(fov_deg, 1.0) -- the camera's own imaging FOV, which
                # has nothing to do with how far off the *hint* itself can
                # be. Confirmed live: a PAA point failed all 5 retries
                # (each a genuinely different, clean, star-rich frame --
                # not a bad-image problem) because the mount's own
                # position hint after park+RA-only rotation reads Dec~90
                # (see AlignmentPanel's own docstring on this), while the
                # true declination was ~87.5 -- a 2.49deg gap, just
                # outside the old ~1.83deg (fov_deg) radius, so solve-
                # field was searching an entirely wrong patch of sky on
                # every attempt. Re-solving the exact same frame offline
                # with --radius 5 found the real match in ~1s (vs. never,
                # at the old radius) -- a wider radius costs essentially
                # nothing here since --scale-low/high already does the
                # real work of narrowing the index search. HINT_RADIUS_DEG
                # is a generous, fixed bound on "how wrong could the
                # mount's own belief be" (pointing error before a PAA/
                # sync fix, not the camera's field of view) -- deliberately
                # NOT derived from fov_deg.
                cmd += ["--ra", str(hint_ra_deg), "--dec", str(hint_dec_deg), "--radius", str(HINT_RADIUS_DEG)]
            cmd.append(str(fits_path))

            env = os.environ.copy()
            shim_root = _ensure_numpy2_shim()
            if shim_root is not None:
                existing = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = str(shim_root) + (os.pathsep + existing if existing else "")

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self._timeout_s, env=env)
            except subprocess.TimeoutExpired:
                return SolveResult(success=False, message=f"solve-field timed out (>{self._timeout_s:.0f}s)")
            except Exception as exc:
                return SolveResult(success=False, message=f"solve-field failed to run: {exc}")

            solved_path = tmp / "result.solved"
            wcs_path = tmp / "result.wcs"
            if not solved_path.exists() or not wcs_path.exists():
                return SolveResult(
                    success=False,
                    message=f"No solution (exit {proc.returncode}). stderr: {proc.stderr[-300:]}",
                )
            return _parse_astrometry_wcs(wcs_path)

    def solve_async(
        self,
        frame: np.ndarray,
        tk_widget,
        on_done,
        fov_deg: float = 1.0,
        hint_ra_deg: float | None = None,
        hint_dec_deg: float | None = None,
    ) -> None:
        """Non-blocking solve -- see PlateSolver.solve_async."""
        _solve_async(self.solve, frame, tk_widget, on_done, fov_deg, hint_ra_deg, hint_dec_deg)


def _solve_async(solve_fn, frame, tk_widget, on_done, fov_deg, hint_ra_deg, hint_dec_deg) -> None:
    def _run() -> None:
        result = solve_fn(frame, fov_deg, hint_ra_deg, hint_dec_deg)
        tk_widget.after(0, lambda: on_done(result))

    threading.Thread(target=_run, daemon=True).start()


def _parse_astrometry_wcs(path: Path) -> SolveResult:
    import math
    from astropy.io import fits

    header = fits.getheader(path)
    try:
        ra_deg = float(header["CRVAL1"])
        dec_deg = float(header["CRVAL2"])
    except KeyError as exc:
        return SolveResult(success=False, message=f"Missing CRVAL1/CRVAL2 in WCS: {exc}")

    try:
        cd11 = float(header.get("CD1_1", 0.0))
        cd12 = float(header.get("CD1_2", 0.0))
        cd21 = float(header.get("CD2_1", 0.0))
        cd22 = float(header.get("CD2_2", 0.0))
        pixel_scale_deg = math.sqrt(cd11 ** 2 + cd21 ** 2)
        rotation_deg = math.degrees(math.atan2(cd12, cd11))
        # See SolveResult.flip_parity's own docstring -- the real parity
        # of THIS solve's optical path, read from the matrix itself, not
        # assumed.
        flip_parity = (cd11 * cd22 - cd12 * cd21) > 0.0
    except Exception:
        pixel_scale_deg = 0.0
        rotation_deg = 0.0
        flip_parity = False

    return SolveResult(
        success=True,
        ra_deg=ra_deg, dec_deg=dec_deg,
        field_rotation_deg=rotation_deg,
        pixel_scale_arcsec=pixel_scale_deg * 3600.0,
        flip_parity=flip_parity,
    )


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
        cd22 = float(kv.get("CD2_2", "0"))
        pixel_scale_deg = math.sqrt(cd11 ** 2 + cd21 ** 2)
        rotation_deg = math.degrees(math.atan2(cd12, cd11))
        # See SolveResult.flip_parity's own docstring.
        flip_parity = (cd11 * cd22 - cd12 * cd21) > 0.0
    except Exception:
        pixel_scale_deg = 0.0
        rotation_deg = 0.0
        flip_parity = False

    return SolveResult(
        success=True,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        field_rotation_deg=rotation_deg,
        pixel_scale_arcsec=pixel_scale_deg * 3600.0,
        flip_parity=flip_parity,
    )
