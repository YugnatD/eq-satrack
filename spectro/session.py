"""One observation session's output directory -- created once (see
Session.ensure, called from ReductionPanel's own "Build masters" step,
by which point both the reference and target stars are normally already
chosen), named from the date and the two stars actually being observed,
so a session on disk is self-describing and reproducible without cross-
referencing anything else. Holds every RAW captured frame (as a stacked
cube, not one file per frame -- 20 darks as one (20,H,W) FITS is both
more practical and still exactly "the raw data", unprocessed), every
master calibration frame, the per-star calibrated/stacked frame, the
final spectrum, and a metadata file recording the instrument/site setup
(grating, focal length, Star Analyser-to-sensor distance, pixel size,
site, dispersion...) -- everything that actually fed the final result
and would be needed to reproduce or re-check it later."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits


def _slug(name: str | None, fallback: str) -> str:
    if not name:
        return fallback
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return cleaned or fallback


class Session:
    def __init__(self, root_dir: Path):
        self._root_dir = root_dir
        self.dir: Path | None = None

    def ensure(self, reference_name: str | None, target_name: str | None) -> Path:
        """Creates the session directory on the FIRST call and reuses it
        for every later call regardless of what's passed then -- named
        once, from whatever star names are resolved at that moment."""
        if self.dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ref = _slug(reference_name, "ref")
            tgt = _slug(target_name, "target")
            self.dir = self._root_dir / f"{stamp}_{tgt}_ref_{ref}"
            self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    def save_fits(self, filename: str, data: np.ndarray) -> None:
        if self.dir is None:
            return
        fits.PrimaryHDU(data=data.astype(np.float32)).writeto(self.dir / filename, overwrite=True)

    def save_fits_cube(self, filename: str, frames: list[np.ndarray]) -> None:
        """The RAW captured frames, stacked into one (N, H, W) FITS file
        rather than N separate files -- still literally the unprocessed
        per-frame data (nothing here is averaged/corrected), just not one
        file per frame, which for e.g. 20 darks would be needless clutter
        for the same information."""
        if self.dir is None or not frames:
            return
        cube = np.stack(frames, axis=0).astype(np.float32)
        fits.PrimaryHDU(data=cube).writeto(self.dir / filename, overwrite=True)

    def save_spectrum_fits(self, filename: str, wl: np.ndarray, flux: np.ndarray) -> None:
        if self.dir is None:
            return
        columns = fits.ColDefs([
            fits.Column(name="wavelength", format="D", unit="Angstrom", array=wl),
            fits.Column(name="flux", format="D", array=flux),
        ])
        fits.BinTableHDU.from_columns(columns).writeto(self.dir / filename, overwrite=True)

    def write_metadata(self, info: dict) -> None:
        """Plain-text key: value dump (nested dicts indented) of the
        instrument/site/target setup used for this session -- not a FITS
        file since this is descriptive/reproducibility info about the
        WHOLE session, not per-frame pixel data."""
        if self.dir is None:
            return
        lines: list[str] = []

        def _write(d: dict, indent: int) -> None:
            for key, value in d.items():
                if isinstance(value, dict):
                    lines.append("  " * indent + f"{key}:")
                    _write(value, indent + 1)
                else:
                    lines.append("  " * indent + f"{key}: {value}")

        _write(info, 0)
        (self.dir / "session_info.txt").write_text("\n".join(lines) + "\n")
