"""Real reference spectra from the Pickles (1998) stellar flux library --
"A Stellar Spectral Flux Library: 1150-25000 A" (Publ. Astron. Soc. Pac.
110, 863), hosted by CDS/VizieR (catalog J/PASP/110/863). 131 flux-
calibrated template spectra (combined from several real observed-spectra
sources, see the catalog's own ReadMe for the source list), one per
spectral type + luminosity class, at 5 A sampling and R~500 resolution --
matches a Star Analyser's own resolution, and is the real, standard
reference data this whole calibration workflow is meant to use (not a
synthetic/model approximation -- see catalog.py's model_spectrum for that
fallback, used only when no reasonable Pickles match exists).

File list below is the actual UVILIB file set from the catalog's real
directory listing (cdsarc.cds.unistra.fr/ftp/J/PASP/110/863/), fetched and
verified while building this module -- not guessed. Deliberately excludes
the "uk" (UVKLIB, extends further into the infrared -- we don't need past
~7200 A), "r"/"w" (metal-rich/metal-weak variants), and the three non-
spectrum files (synphot.dat, lew.dat, irstphot.dat).
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

CATALOG_BASE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/PASP/110/863"
FETCH_TIMEOUT_S = 20

_TEMPLATE_NAMES = [
    "o5v", "o8iii", "o9v",
    "b0i", "b0v", "b12iii", "b1i", "b1v", "b2ii", "b2iv", "b3i", "b3iii", "b3v",
    "b57v", "b5i", "b5ii", "b5iii", "b6iv", "b8i", "b8v", "b9iii", "b9v",
    "a0i", "a0iii", "a0iv", "a0v", "a2i", "a2v", "a3iii", "a3v", "a47iv",
    "a5iii", "a5v", "a7iii", "a7v",
    "f02iv", "f0i", "f0ii", "f0iii", "f0v", "f2ii", "f2iii", "f2v",
    "f5i", "f5iii", "f5iv", "f5v", "f6v", "f8i", "f8iv", "f8v",
    "g0i", "g0iii", "g0iv", "g0v", "g2i", "g2iv", "g2v",
    "g5i", "g5ii", "g5iii", "g5iv", "g5v", "g8i", "g8iii", "g8iv", "g8v",
    "k01ii", "k0iii", "k0iv", "k0v", "k1iii", "k1iv", "k2i", "k2iii", "k2v",
    "k34ii", "k3i", "k3iii", "k3iv", "k3v", "k4i", "k4iii", "k4v", "k5iii", "k5v", "k7v",
    "m0iii", "m0v", "m10iii", "m1iii", "m1v", "m2i", "m2iii", "m2p5v", "m2v",
    "m3ii", "m3iii", "m3v", "m4iii", "m4v", "m5iii", "m5v", "m6iii", "m6v",
    "m7iii", "m8iii", "m9iii",
]

# Luminosity-class roman numeral -> a numeric "size" rank (I=supergiant,
# smallest number here = most luminous/largest star), used only to pick
# the closest AVAILABLE template when there's no exact match -- e.g.
# Regulus is B8IVn (subgiant) but the library has no b8iv, so this picks
# b8v (rank distance 1) over b8i (rank distance 3).
_LUMCLASS_RANK = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}


@dataclass(frozen=True)
class PicklesTemplate:
    name: str  # e.g. "a0v"
    letter: str
    subtype: float
    lumclass_rank: int


def _parse_template_name(name: str) -> PicklesTemplate:
    letter = name[0].upper()
    rest = name[1:]
    digits = ""
    i = 0
    while i < len(rest) and (rest[i].isdigit() or rest[i] == "."):
        digits += rest[i]
        i += 1
    lum_code = rest[i:].upper()
    rank = _LUMCLASS_RANK.get(lum_code, 5)  # unrecognized/blank -- assume dwarf, the most common case
    return PicklesTemplate(name=name, letter=letter, subtype=float(digits) if digits else 5.0, lumclass_rank=rank)


_TEMPLATES = [_parse_template_name(n) for n in _TEMPLATE_NAMES]


def _parse_query_sptype(spectral_type: str) -> tuple[str, float, int] | None:
    if not spectral_type:
        return None
    letter = spectral_type[0].upper()
    if letter not in {"O", "B", "A", "F", "G", "K", "M"}:
        return None
    rest = spectral_type[1:]
    digits = ""
    i = 0
    while i < len(rest) and (rest[i].isdigit() or rest[i] == "."):
        digits += rest[i]
        i += 1
    subtype = float(digits) if digits else 5.0
    lum_code = ""
    for ch in rest[i:]:
        if ch.upper() in "IV":
            lum_code += ch.upper()
        else:
            break
    rank = _LUMCLASS_RANK.get(lum_code, 5)
    return letter, subtype, rank


def find_best_template(spectral_type: str) -> str | None:
    """Closest available Pickles template name for a SIMBAD spectral type
    string, or None if the leading letter isn't O/B/A/F/G/K/M. Matches
    within the same letter only (never substitutes a different spectral
    class); among same-letter templates, weighs subtype distance more
    than luminosity-class distance (temperature dominates the continuum
    shape far more than luminosity class does)."""
    parsed = _parse_query_sptype(spectral_type)
    if parsed is None:
        return None
    letter, subtype, lumclass_rank = parsed
    same_letter = [t for t in _TEMPLATES if t.letter == letter]
    if not same_letter:
        return None
    best = min(same_letter, key=lambda t: abs(t.subtype - subtype) + 0.5 * abs(t.lumclass_rank - lumclass_rank))
    return best.name


class FetchError(RuntimeError):
    pass


_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def fetch_template_spectrum(template_name: str) -> tuple[np.ndarray, np.ndarray]:
    """(wavelength_angstrom, normalized_flux) for a Pickles template,
    e.g. "a0v" -- downloads and gunzips the real .dat.gz file from CDS
    (cached after the first fetch). Raises FetchError on any network/
    parse problem rather than silently falling back -- callers decide
    whether to fall back to the synthetic model (see catalog.py)."""
    if template_name in _cache:
        return _cache[template_name]
    url = f"{CATALOG_BASE_URL}/{template_name}.dat.gz"
    try:
        with urlopen(url, timeout=FETCH_TIMEOUT_S) as response:  # noqa: S310 - fixed https CDS host, not user input
            raw = response.read()
    except (URLError, TimeoutError, OSError) as exc:
        raise FetchError(f"could not fetch {url}: {exc}") from exc
    try:
        text = gzip.decompress(raw).decode("ascii")
    except OSError as exc:
        raise FetchError(f"could not decompress {url}: {exc}") from exc
    wl, flux = [], []
    for line in io.StringIO(text):
        parts = line.split()
        if len(parts) < 2:
            continue
        wl.append(float(parts[0]))
        flux.append(float(parts[1]))
    if not wl:
        raise FetchError(f"{url} parsed to zero data rows")
    wl_arr, flux_arr = np.array(wl), np.array(flux)
    _cache[template_name] = (wl_arr, flux_arr)
    return wl_arr, flux_arr


def fetch_reference_spectrum(
    spectral_type: str, wl_min: float = 3800.0, wl_max: float = 7200.0,
) -> tuple[str, np.ndarray, np.ndarray] | None:
    """(template_name, wavelength_angstrom, normalized_flux) for the best
    real Pickles template matching `spectral_type`, cropped to
    [wl_min, wl_max]. None if the spectral type doesn't parse to a known
    letter. Raises FetchError (not None) on a network/parse failure, so
    callers can distinguish "no sensible template" from "the fetch
    itself failed" and report each differently."""
    template_name = find_best_template(spectral_type)
    if template_name is None:
        return None
    wl, flux = fetch_template_spectrum(template_name)
    mask = (wl >= wl_min) & (wl <= wl_max)
    cropped_wl, cropped_flux = wl[mask], flux[mask]
    if cropped_flux.max() > 0:
        cropped_flux = cropped_flux / cropped_flux.max()
    return template_name, cropped_wl, cropped_flux
