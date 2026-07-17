"""A small, hand-picked list of bright, well-known named stars with their
real J2000 coordinates -- for a "GOTO a star" picker (see
am5/gui/jog_window.py) and the Alignment tab's multi-star sync sky map
(see am5/gui/panels.py's visible_named_stars), where a human-recognizable
name is far more usable than browsing tens of thousands of Hipparcos
entries by number.

Coordinates verified directly against real catalogues, not typed from
memory -- (ra_deg, dec_deg, magnitude), J2000. The original ~30 entries
came from the Hipparcos catalogue (the same source as assets/
bright_stars.npz); the additional entries down to Vmag ~2.2 came from the
official IAU Catalog of Star Names (IAU-CSN, IAU Working Group on Star
Names -- https://www.pas.rochester.edu/~emamajek/WGSN/IAU-CSN.txt),
picking only IAU-approved proper names bright/well-known enough to
identify at the eyepiece, same reasoning as visible_named_stars' own
docstring on why this list stays small and curated rather than pulling
from the much larger anonymous catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NamedStar:
    name: str
    ra_deg: float
    dec_deg: float
    magnitude: float

    @property
    def ra_hours(self) -> float:
        return self.ra_deg / 15.0


NAMED_STARS: list[NamedStar] = [
    NamedStar("Sirius", 101.288541, -16.713143, -1.44),
    NamedStar("Canopus", 95.987878, -52.695718, -0.62),
    NamedStar("Alpha Centauri", 219.920410, -60.835147, -0.01),
    NamedStar("Arcturus", 213.918114, 19.187270, -0.05),
    NamedStar("Vega", 279.234108, 38.782993, 0.03),
    NamedStar("Capella", 79.172065, 45.999029, 0.08),
    NamedStar("Rigel", 78.634464, -8.201639, 0.18),
    NamedStar("Procyon", 114.827242, 5.227508, 0.40),
    NamedStar("Betelgeuse", 88.792872, 7.407036, 0.45),
    NamedStar("Achernar", 24.428132, -57.236660, 0.45),
    NamedStar("Hadar", 210.956019, -60.372978, 0.61),
    NamedStar("Altair", 297.694509, 8.867385, 0.76),
    NamedStar("Spica", 201.298352, -11.161245, 0.98),
    NamedStar("Aldebaran", 68.980002, 16.509762, 0.87),
    NamedStar("Antares", 247.351948, -26.431946, 1.06),
    NamedStar("Pollux", 116.330683, 28.026310, 1.16),
    NamedStar("Fomalhaut", 344.411773, -29.621837, 1.17),
    NamedStar("Deneb", 310.357973, 45.280334, 1.25),
    NamedStar("Regulus", 152.093581, 11.967195, 1.36),
    NamedStar("Castor", 113.650019, 31.888636, 1.58),
    NamedStar("Bellatrix", 81.282784, 6.349735, 1.64),
    NamedStar("Alnilam", 84.053386, -1.201917, 1.69),
    NamedStar("Alnitak", 85.189687, -1.942578, 1.74),
    NamedStar("Alioth", 193.506804, 55.959843, 1.76),
    NamedStar("Dubhe", 165.932654, 61.751119, 1.81),
    NamedStar("Alkaid", 206.885609, 49.313303, 1.85),
    NamedStar("Polaris", 37.946147, 89.264138, 1.97),
    NamedStar("Saiph", 86.939116, -9.669602, 2.07),
    NamedStar("Kochab", 222.676648, 74.155476, 2.07),
    NamedStar("Mizar", 200.980916, 54.925415, 2.23),
    NamedStar("Mintaka", 83.001666, -0.299093, 2.25),
    # Added for better sky coverage in the Alignment tab's multi-star sync
    # (more well-spread candidate points at any given time/location) --
    # see module docstring for the IAU-CSN source.
    NamedStar("Adhara", 104.656453, -28.972086, 1.50),
    NamedStar("Miaplacidus", 138.299906, -69.717208, 1.67),
    NamedStar("Alnair", 332.058270, -46.960974, 1.73),
    NamedStar("Mirfak", 51.080709, 49.861179, 1.79),
    NamedStar("Kaus Australis", 276.042993, -34.384616, 1.79),
    NamedStar("Wezen", 107.097850, -26.393200, 1.83),
    NamedStar("Sargas", 264.329711, -42.997824, 1.86),
    NamedStar("Avior", 125.628480, -59.509484, 1.86),
    NamedStar("Menkalinan", 89.882179, 44.947433, 1.90),
    NamedStar("Atria", 252.166229, -69.027712, 1.91),
    NamedStar("Alhena", 99.427960, 16.399280, 1.93),
    NamedStar("Peacock", 306.411904, -56.735090, 1.94),
    NamedStar("Mirzam", 95.674939, -17.955919, 1.98),
    NamedStar("Alphard", 141.896847, -8.658602, 1.99),
    NamedStar("Diphda", 10.897379, -17.986606, 2.04),
    NamedStar("Nunki", 283.816360, -26.296724, 2.05),
    NamedStar("Menkent", 211.670617, -36.369958, 2.06),
    NamedStar("Alpheratz", 2.096916, 29.090431, 2.07),
    NamedStar("Mirach", 17.433013, 35.620557, 2.07),
    NamedStar("Rasalhague", 263.733627, 12.560035, 2.08),
    NamedStar("Shaula", 263.402167, -37.103824, 2.08),
    NamedStar("Denebola", 177.264910, 14.572058, 2.14),
    NamedStar("Naos", 120.896031, -40.003148, 2.21),
]

NAMED_STARS_BY_NAME: dict[str, NamedStar] = {s.name: s for s in NAMED_STARS}
