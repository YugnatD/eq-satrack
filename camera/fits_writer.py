"""Single-frame FITS snapshots (reference frame, dark, flat) — not the video
sequence, that's camera/ser_writer.py's job. Thin wrapper over astropy.io.fits,
already a dependency (astropy is installed for other reasons on this machine)
rather than a hand-rolled writer, unlike SER where no such library was at hand.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits


def write_fits(path: Path, frame: np.ndarray, header_extra: dict[str, object] | None = None) -> None:
    header = fits.Header()
    header["DATE-OBS"] = datetime.now(timezone.utc).isoformat()
    for key, value in (header_extra or {}).items():
        header[key] = value
    hdu = fits.PrimaryHDU(data=frame, header=header)
    hdu.writeto(Path(path), overwrite=True)
