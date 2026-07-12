"""Bright-star stick-figure data for a handful of well-known constellations,
for drawing a recognizable sky-chart background -- not a full star atlas.

Coordinates are approximate J2000 catalog values (RA hours, DEC degrees) for
each constellation's brightest stars, hand-entered from standard bright-star
data (Bayer-designated stars only) -- fine for a schematic chart, not
precise astrometry. Picked for visibility from mid-northern latitudes
(circumpolar Ursa Major/Minor/Cassiopeia, plus seasonal Cygnus and Orion).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from skyfield.toposlib import GeographicPosition

from .angles import equatorial_to_altaz


@dataclass(frozen=True)
class ConstellationShape:
    name: str
    stars_radec: list[tuple[float, float]]  # (ra_hours, dec_deg), J2000
    lines: list[tuple[int, int]]  # index pairs into stars_radec


CONSTELLATIONS: list[ConstellationShape] = [
    ConstellationShape(
        name="Ursa Major",
        stars_radec=[
            (11 + 3 / 60 + 43.7 / 3600, 61 + 45 / 60 + 4 / 3600),  # Dubhe
            (11 + 1 / 60 + 50.5 / 3600, 56 + 22 / 60 + 57 / 3600),  # Merak
            (11 + 53 / 60 + 49.8 / 3600, 53 + 41 / 60 + 41 / 3600),  # Phecda
            (12 + 15 / 60 + 25.6 / 3600, 57 + 1 / 60 + 57 / 3600),  # Megrez
            (12 + 54 / 60 + 1.7 / 3600, 55 + 57 / 60 + 35 / 3600),  # Alioth
            (13 + 23 / 60 + 55.5 / 3600, 54 + 55 / 60 + 31 / 3600),  # Mizar
            (13 + 47 / 60 + 32.4 / 3600, 49 + 18 / 60 + 48 / 3600),  # Alkaid
        ],
        lines=[(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 6)],
    ),
    ConstellationShape(
        name="Ursa Minor",
        stars_radec=[
            (2 + 31 / 60 + 49.1 / 3600, 89 + 15 / 60 + 51 / 3600),  # Polaris
            (17 + 32 / 60 + 13.0 / 3600, 86 + 35 / 60 + 11 / 3600),  # Yildun
            (16 + 45 / 60 + 58.0 / 3600, 82 + 2 / 60 + 14 / 3600),  # eps UMi
            (15 + 44 / 60 + 3.5 / 3600, 77 + 47 / 60 + 40 / 3600),  # zeta UMi
            (14 + 50 / 60 + 42.3 / 3600, 74 + 9 / 60 + 20 / 3600),  # Kochab
            (15 + 20 / 60 + 43.7 / 3600, 71 + 50 / 60 + 2 / 3600),  # Pherkad
            (16 + 17 / 60 + 30.4 / 3600, 75 + 45 / 60 + 19 / 3600),  # eta UMi
        ],
        lines=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 3)],
    ),
    ConstellationShape(
        name="Cassiopeia",
        stars_radec=[
            (0 + 9 / 60 + 10.7 / 3600, 59 + 8 / 60 + 59 / 3600),  # Caph
            (0 + 40 / 60 + 30.4 / 3600, 56 + 32 / 60 + 14 / 3600),  # Shedar
            (0 + 56 / 60 + 42.5 / 3600, 60 + 43 / 60),  # Tsih
            (1 + 25 / 60 + 49.0 / 3600, 60 + 14 / 60 + 7 / 3600),  # Ruchbah
            (1 + 54 / 60 + 23.7 / 3600, 63 + 40 / 60 + 12 / 3600),  # Segin
        ],
        lines=[(0, 1), (1, 2), (2, 3), (3, 4)],
    ),
    ConstellationShape(
        name="Cygnus",
        stars_radec=[
            (20 + 41 / 60 + 25.9 / 3600, 45 + 16 / 60 + 49 / 3600),  # Deneb
            (20 + 22 / 60 + 13.7 / 3600, 40 + 15 / 60 + 24 / 3600),  # Sadr
            (20 + 46 / 60 + 12.7 / 3600, 33 + 58 / 60 + 13 / 3600),  # Gienah
            (19 + 44 / 60 + 58.5 / 3600, 45 + 7 / 60 + 51 / 3600),  # delta Cyg
            (19 + 30 / 60 + 43.3 / 3600, 27 + 57 / 60 + 35 / 3600),  # Albireo
        ],
        lines=[(0, 1), (1, 2), (3, 1), (1, 4)],
    ),
    ConstellationShape(
        name="Orion",
        stars_radec=[
            (5 + 55 / 60 + 10.3 / 3600, 7 + 24 / 60 + 25 / 3600),  # Betelgeuse
            (5 + 25 / 60 + 7.9 / 3600, 6 + 20 / 60 + 59 / 3600),  # Bellatrix
            (5 + 32 / 60 + 0.4 / 3600, -0 - 17 / 60 - 57 / 3600),  # Mintaka
            (5 + 36 / 60 + 12.8 / 3600, -1 - 12 / 60 - 7 / 3600),  # Alnilam
            (5 + 40 / 60 + 45.5 / 3600, -1 - 56 / 60 - 34 / 3600),  # Alnitak
            (5 + 47 / 60 + 45.4 / 3600, -9 - 40 / 60 - 11 / 3600),  # Saiph
            (5 + 14 / 60 + 32.3 / 3600, -8 - 12 / 60 - 6 / 3600),  # Rigel
        ],
        lines=[(0, 1), (0, 4), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6)],
    ),
]


@dataclass(frozen=True)
class ConstellationAltAz:
    name: str
    stars_azalt: list[tuple[float, float]]  # (az_deg, alt_deg), same order/index as ConstellationShape.stars_radec
    lines: list[tuple[int, int]]


def constellations_altaz(site: GeographicPosition, when: datetime) -> list[ConstellationAltAz]:
    """Alt/az for every star in CONSTELLATIONS, evaluated at a single
    reference time `when` -- a static snapshot of the background sky (real
    star motion over a several-minute pass is imperceptible), not
    recomputed per-trajectory-sample.

    Uses direct spherical trig (am5.angles.equatorial_to_altaz), not
    Skyfield's Star/.observe() -- that path needs a loaded JPL ephemeris
    (de421.bsp) for light-time/aberration, which this project doesn't
    otherwise depend on. Stars' parallax/light-time from Earth's surface is
    negligible for a schematic chart, so the same GMST-based trig already
    used (and verified) for the mock mount's own alt/az is enough here."""
    lat_deg = site.latitude.degrees
    lon_deg = site.longitude.degrees
    results = []
    for shape in CONSTELLATIONS:
        stars_azalt = [
            equatorial_to_altaz(ra_hours * 15.0, dec_deg, lat_deg, lon_deg, when)
            for ra_hours, dec_deg in shape.stars_radec
        ]
        results.append(ConstellationAltAz(name=shape.name, stars_azalt=stars_azalt, lines=shape.lines))
    return results
