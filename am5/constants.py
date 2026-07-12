"""Physical constants shared by the mock, the protocol layer and the analysis code."""

# Degrees of sky rotation per second of real time (1x sidereal), i.e. 15.041"/s.
# Derived from the sidereal day (23h56m4.0905s = 86164.0905 s / 360 deg), expressed
# the more common way as 360.98564736629 deg per 86400 SI seconds.
SIDEREAL_DEG_PER_S = 360.98564736629 / 86400.0
