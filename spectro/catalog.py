"""Real star lookup (SIMBAD, via astroquery) and spectrophotometric
standard-star candidate selection. Pure functions, no Tk -- the GUI layer
(spectro/gui/panels.py) runs these on a background thread (network I/O,
can take seconds or fail) and posts results back through a queue, the
same pattern am5/gui/panels.py's PassesPanel uses for its own background
TLE/trajectory work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astroquery.simbad import Simbad

SIMBAD_TIMEOUT_S = 15

# A0V-A3V dwarfs/giants are the amateur-spectroscopy convention for a flux
# standard (well-characterized, mostly-featureless blue continuum) -- see
# the ARAS/ISIS practice discussed when this module was designed. Vmag
# cutoff keeps candidates bright enough for a small amateur setup.
_STANDARD_SPTYPE_PREFIXES = ("A0", "A1", "A2", "A3")
_STANDARD_VMAG_LIMIT = 6.5
_STANDARD_SEARCH_RADIUS_DEG = 20.0
_STANDARD_MAX_RESULTS = 6

_simbad = Simbad()
_simbad.TIMEOUT = SIMBAD_TIMEOUT_S
_simbad.add_votable_fields("sp_type", "flux(V)")


class StarNotFound(RuntimeError):
    pass


@dataclass
class Star:
    name: str
    ra_deg: float
    dec_deg: float
    vmag: float | None
    spectral_type: str


@dataclass
class StandardCandidate:
    star: Star
    separation_deg: float
    airmass_delta: float | None  # None if either star is below the horizon at `when`


def resolve_target(name: str) -> Star:
    """Real SIMBAD name resolution -- raises StarNotFound if SIMBAD has no
    match, not a guessed fallback. Network call: expect ~1-3s."""
    result = _simbad.query_object(name)
    if result is None or len(result) == 0:
        raise StarNotFound(f"SIMBAD has no match for {name!r}")
    row = result[0]
    vmag = float(row["V"]) if row["V"] is not None else None
    return Star(
        name=str(row["main_id"]), ra_deg=float(row["ra"]), dec_deg=float(row["dec"]),
        vmag=vmag, spectral_type=str(row["sp_type"] or "").strip(),
    )


def is_standard_candidate(star: Star) -> bool:
    """True if `star` itself already meets the flux-standard criteria
    (bright A0-A3) used by find_standard_candidates below -- e.g. Vega,
    which is A0V and Vmag 0.03. When the TARGET is one of these, it
    doesn't need a companion standard at all: it already fills that role
    for itself, see TargetPanel's handling of this case."""
    if star.vmag is None or star.vmag >= _STANDARD_VMAG_LIMIT:
        return False
    return star.spectral_type.startswith(_STANDARD_SPTYPE_PREFIXES)


# Rough main-sequence Teff by spectral subtype (Pecaut & Mamajek dwarf
# scale, a handful of anchor points) -- NOT luminosity-class aware (a
# giant/supergiant of the same letter+number runs cooler than a dwarf), so
# this is only ever used for "what continuum shape is plausible", never
# presented as a measured stellar parameter. Encoded as a single numeric
# axis (10 units per spectral letter) so interpolation is a single
# np.interp call across letter boundaries.
_SPTYPE_LETTER_BASE = {"O": 0.0, "B": 10.0, "A": 20.0, "F": 30.0, "G": 40.0, "K": 50.0, "M": 60.0}
_SPTYPE_TEFF_ANCHORS = [
    ("O5", 42000), ("O9", 33000), ("B0", 30000), ("B3", 19000), ("B5", 15400), ("B8", 11900),
    ("A0", 9700), ("A2", 9000), ("A5", 8200), ("A7", 7850), ("F0", 7200), ("F5", 6650),
    ("G0", 5940), ("G2", 5770), ("G5", 5610), ("K0", 5240), ("K5", 4410), ("M0", 3870), ("M5", 3170),
]
_TEFF_TABLE_X = [_SPTYPE_LETTER_BASE[code[0]] + float(code[1:]) for code, _ in _SPTYPE_TEFF_ANCHORS]
_TEFF_TABLE_Y = [teff for _, teff in _SPTYPE_TEFF_ANCHORS]

# Real rest wavelengths (air, Angstrom) -- the same lines a Star Analyser
# (R~100-500) can plausibly resolve, used to mark where absorption
# features are EXPECTED on the model continuum below (not synthesized
# dips -- no real equivalent-width data backs a depth/width guess).
REFERENCE_LINES = [
    ("Ca II K", 3933.66), ("Ca II H", 3968.47), ("Hδ", 4101.73), ("Hγ", 4340.46),
    ("Hβ", 4861.35), ("Na D", 5892.9), ("Hα", 6562.79),
]


def estimate_teff_k(spectral_type: str) -> float | None:
    """Rough Teff from a SIMBAD spectral type string (e.g. "A2Ia",
    "B8IVn", "G0V") -- letter + leading digit only, luminosity class
    ignored, linearly interpolated against _SPTYPE_TEFF_ANCHORS. None if
    the leading letter isn't O/B/A/F/G/K/M (e.g. a white dwarf "DA" or a
    variable-star designation with no clean spectral type)."""
    if not spectral_type:
        return None
    letter = spectral_type[0].upper()
    if letter not in _SPTYPE_LETTER_BASE:
        return None
    digits = ""
    for ch in spectral_type[1:]:
        if ch.isdigit() or ch == ".":
            digits += ch
        else:
            break
    subtype = float(digits) if digits else 5.0
    x = _SPTYPE_LETTER_BASE[letter] + subtype
    return float(np.interp(x, _TEFF_TABLE_X, _TEFF_TABLE_Y))


_HC_OVER_K_ANGSTROM_K = 1.4387769e8  # hc/k, in angstrom*kelvin


def model_spectrum(teff_k: float, wl_min: float = 3800.0, wl_max: float = 7200.0, n: int = 900) -> tuple[np.ndarray, np.ndarray]:
    """A blackbody continuum at teff_k (real Planck law in wavelength
    space, B_lambda ~ 1/lambda^5 / (exp(hc/lambda k T) - 1) -- peaks at
    Wien's wavelength and falls off on BOTH sides, unlike a first attempt
    at this function that reused the frequency-domain shape x^3/(e^x-1)
    plotted directly against wavelength: that shape doesn't have a peak
    in this range, so it rendered as a near-straight line instead of the
    curved, peaked shape a real continuum has -- confirmed wrong by
    checking a real hot star's (Vega, 9700K) plotted range against where
    Wien's law actually puts the peak, ~2990 Angstrom, blueward of the
    whole 3800-7200 plotted window, so the visible-range shape should be
    monotonically FALLING toward red, which this corrected version does).

    Still NOT a real stellar atmosphere model (no line physics, no real
    equivalent widths) and NOT a measured reference spectrum -- only
    stands in for "roughly what shape should this star's continuum have"
    when no archival spectrum is available; always labeled as a model
    everywhere it's plotted (see TargetPanel). REFERENCE_LINES above
    still marks real rest wavelengths on top of it, since those don't
    depend on this approximation."""
    wl = np.linspace(wl_min, wl_max, n)
    x = _HC_OVER_K_ANGSTROM_K / (wl * teff_k)
    flux = (1.0 / wl**5) / (np.exp(np.clip(x, 1e-3, 500)) - 1.0)
    return wl, flux / flux.max()


def _airmass(coord: SkyCoord, frame: AltAz) -> float | None:
    altaz = coord.transform_to(frame)
    if altaz.alt.deg <= 0:
        return None  # below the horizon -- not usable right now
    return float(altaz.secz)


def find_standard_candidates(
    target: Star, site_lat_deg: float, site_lon_deg: float, site_elevation_m: float,
    when: datetime | None = None,
) -> list[StandardCandidate]:
    """Real SIMBAD region query around the target for bright A0-A3 stars,
    ranked by angular separation AND airmass match at `when` (default:
    now) -- both matter for how well the standard's correction transfers
    to the target, see TargetPanel's own explanation label. Network call:
    expect a few seconds. Returns [] (not an exception) if SIMBAD has
    nothing suitable in range -- a real, if inconvenient, possibility for
    a sparse patch of sky."""
    when = when or datetime.now(timezone.utc)
    sptype_clause = " OR ".join(f"basic.sp_type LIKE '{prefix}%'" for prefix in _STANDARD_SPTYPE_PREFIXES)
    query = f"""
    SELECT basic.main_id, basic.ra, basic.dec, basic.sp_type, flux.flux AS vmag
    FROM basic
    JOIN flux ON flux.oidref = basic.oid
    WHERE flux.filter = 'V'
      AND flux.flux < {_STANDARD_VMAG_LIMIT}
      AND ({sptype_clause})
      AND CONTAINS(POINT('ICRS', basic.ra, basic.dec),
                   CIRCLE('ICRS', {target.ra_deg}, {target.dec_deg}, {_STANDARD_SEARCH_RADIUS_DEG})) = 1
    ORDER BY vmag ASC
    """
    table = _simbad.query_tap(query)
    if table is None or len(table) == 0:
        return []

    location = EarthLocation(lat=site_lat_deg * u.deg, lon=site_lon_deg * u.deg, height=site_elevation_m * u.m)
    frame = AltAz(obstime=Time(when), location=location)
    target_coord = SkyCoord(ra=target.ra_deg * u.deg, dec=target.dec_deg * u.deg)
    target_airmass = _airmass(target_coord, frame)

    candidates = []
    for row in table:
        star = Star(
            name=str(row["main_id"]), ra_deg=float(row["ra"]), dec_deg=float(row["dec"]),
            vmag=float(row["vmag"]), spectral_type=str(row["sp_type"] or "").strip(),
        )
        coord = SkyCoord(ra=star.ra_deg * u.deg, dec=star.dec_deg * u.deg)
        separation_deg = target_coord.separation(coord).deg
        if star.name == target.name or separation_deg < 0.05:
            continue  # the target itself matched its own spectral-type/magnitude filter -- not a real candidate
        airmass = _airmass(coord, frame)
        airmass_delta = abs(airmass - target_airmass) if (airmass is not None and target_airmass is not None) else None
        candidates.append(StandardCandidate(star=star, separation_deg=separation_deg, airmass_delta=airmass_delta))

    # Airmass match weighted heavily (0.1 of airmass delta matters much
    # more than 1 degree of separation) -- a candidate with no airmass
    # (below horizon right now) sorts last, not first, even if very close.
    candidates.sort(key=lambda c: c.separation_deg + (c.airmass_delta if c.airmass_delta is not None else 999.0) * 10.0)
    return candidates[:_STANDARD_MAX_RESULTS]


def altitude_track(
    star: Star, site_lat_deg: float, site_lon_deg: float, site_elevation_m: float,
    when: datetime, hours_span: tuple[float, float] = (-1.0, 4.0), samples: int = 40,
) -> tuple[list[float], list[float]]:
    """(az_deg, alt_deg) over hours_span[0]..hours_span[1] hours relative
    to `when` -- real astropy AltAz transform, for TargetPanel's sky
    chart. Not a full-night ephemeris (no rise/set search), just enough
    to show whether the target and a chosen standard track each other
    reasonably over a plausible observing window."""
    location = EarthLocation(lat=site_lat_deg * u.deg, lon=site_lon_deg * u.deg, height=site_elevation_m * u.m)
    coord = SkyCoord(ra=star.ra_deg * u.deg, dec=star.dec_deg * u.deg)
    offsets_h = [hours_span[0] + (hours_span[1] - hours_span[0]) * i / (samples - 1) for i in range(samples)]
    times = Time([when + timedelta(hours=h) for h in offsets_h])
    altaz = coord.transform_to(AltAz(obstime=times, location=location))
    return list(altaz.az.deg), list(altaz.alt.deg)
