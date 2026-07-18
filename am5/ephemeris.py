"""TLE loading and satellite pass geometry via Skyfield. No mount
dependency — everything here is pure astronomy, testable without hardware
or network.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from skyfield.api import EarthSatellite, load
from skyfield.toposlib import GeographicPosition

from .angles import unwrap_deg
from .optics import estimate_iss_magnitude

CELESTRAK_URL_TEMPLATE = "https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=tle"
ISS_CATNR = 25544
CELESTRAK_ISS_URL = CELESTRAK_URL_TEMPLATE.format(catnr=ISS_CATNR)

# Same endpoint family as CELESTRAK_URL_TEMPLATE, but keyed by GROUP (a
# named collection of NORAD IDs Celestrak curates) instead of a single
# CATNR -- used for PassesPanel's "Live now" sub-tab (am5/gui/panels.py),
# which needs many candidate satellites to check against the current sky,
# not one specific one. VISUAL_GROUP verified live (WebFetch): returns
# ~250 satellites in the same 3-line-per-satellite TLE format
# load.tle_file() already parses for the single-satellite case below.
CELESTRAK_GROUP_URL_TEMPLATE = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
VISUAL_GROUP = "visual"


def _age_hours(path: Path) -> float:
    return (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600.0


def load_satellite_tle(catnr: int, cache_path: Path, max_age_hours: float = 24.0) -> EarthSatellite:
    """Load a satellite's TLE (by NORAD catalog number) from `cache_path`,
    refetching from Celestrak if the cache is missing or older than
    `max_age_hours`. Falls back to a stale cache (with a warning) if the
    network fetch fails, since a slightly outdated TLE beats refusing to
    track at all. Use a distinct `cache_path` per satellite -- reusing one
    across different catalog numbers would silently keep serving whichever
    satellite happened to be cached first."""
    url = CELESTRAK_URL_TEMPLATE.format(catnr=catnr)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    force_reload = not cache_path.exists() or _age_hours(cache_path) > max_age_hours
    ts = load.timescale()
    try:
        satellites = load.tle_file(url, reload=force_reload, filename=str(cache_path), ts=ts)
    except Exception as exc:
        if not cache_path.exists():
            raise RuntimeError(f"no cached TLE and fetch failed: {exc}") from exc
        print(f"[warn] TLE fetch failed ({exc}); using cache from {_age_hours(cache_path):.1f}h ago", file=sys.stderr)
        satellites = load.tle_file(url, reload=False, filename=str(cache_path), ts=ts)
    if not satellites:
        raise RuntimeError(f"no satellites parsed from {cache_path} (invalid NORAD catalog number?)")
    if _age_hours(cache_path) > max_age_hours:
        print(f"[warn] TLE is {_age_hours(cache_path):.1f}h old (> {max_age_hours}h) — orbit-changing "
              f"maneuvers can invalidate it, expect pointing error", file=sys.stderr)
    return satellites[0]


def load_iss_tle(cache_path: Path, max_age_hours: float = 24.0) -> EarthSatellite:
    """ISS-specific convenience wrapper over load_satellite_tle."""
    return load_satellite_tle(ISS_CATNR, cache_path, max_age_hours)


def load_satellite_group_tles(group: str, cache_path: Path, max_age_hours: float = 24.0) -> list[EarthSatellite]:
    """Same cache-first/fetch-fallback logic as load_satellite_tle, but for
    a whole named Celestrak GROUP (e.g. VISUAL_GROUP) instead of one
    CATNR -- returns every satellite in the group instead of just the
    first one. Used by PassesPanel's "Live now" sub-tab (am5/gui/panels.py)
    to check many candidate satellites against the current sky at once,
    rather than tracking one specific known satellite. Use a distinct
    cache_path per group, same reasoning as load_satellite_tle's own
    docstring for per-CATNR cache files."""
    url = CELESTRAK_GROUP_URL_TEMPLATE.format(group=group)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    force_reload = not cache_path.exists() or _age_hours(cache_path) > max_age_hours
    ts = load.timescale()
    try:
        satellites = load.tle_file(url, reload=force_reload, filename=str(cache_path), ts=ts)
    except Exception as exc:
        if not cache_path.exists():
            raise RuntimeError(f"no cached TLE group and fetch failed: {exc}") from exc
        print(f"[warn] TLE group fetch failed ({exc}); using cache from {_age_hours(cache_path):.1f}h ago", file=sys.stderr)
        satellites = load.tle_file(url, reload=False, filename=str(cache_path), ts=ts)
    if not satellites:
        raise RuntimeError(f"no satellites parsed from {cache_path} (invalid group name?)")
    if _age_hours(cache_path) > max_age_hours:
        print(f"[warn] TLE group is {_age_hours(cache_path):.1f}h old (> {max_age_hours}h) — orbit-changing "
              f"maneuvers can invalidate individual entries, expect pointing error", file=sys.stderr)
    return satellites


def currently_visible_satellites(
    satellites: list[EarthSatellite], site: GeographicPosition, horizon_deg: float = 10.0,
    t0: datetime | None = None,
) -> list[tuple[EarthSatellite, float, float]]:
    """(satellite, alt_deg, az_deg) for every satellite in `satellites`
    currently above `horizon_deg` at `t0` (now, if not given), sorted by
    altitude descending (highest first -- the easiest, most centrally
    overhead target to point at right now). Pure SGP4 propagation, no
    network -- safe to call from a background thread against an
    already-loaded satellite list (see PassesPanel's "Live now" sub-tab,
    am5/gui/panels.py, which loads the group once via
    load_satellite_group_tles and calls this repeatedly on a timer)."""
    ts = load.timescale()
    t0 = t0 or datetime.now(timezone.utc)
    t = ts.from_datetime(t0)
    visible = []
    for sat in satellites:
        # Regression fix, found by code audit: this used to have no
        # per-satellite fault isolation -- a real Celestrak "visual" group
        # fetch (~150-250 entries) can include a malformed/degenerate TLE,
        # and any exception here used to abort the WHOLE scan, which
        # propagates up through PassesPanel's "Live now" refresh handlers
        # as a generic "live_error", leaving the entire list empty/erroring
        # on every subsequent auto-refresh tick until a manual catalog
        # reload happens to no longer include the offending entry. One bad
        # satellite should never take down the whole live view.
        try:
            alt, az, _ = (sat - site).at(t).altaz()
        except Exception as exc:  # noqa: BLE001 - isolate one bad satellite, don't abort the whole scan
            print(f"[warn] skipping {sat.name!r} in currently_visible_satellites: {exc}", file=sys.stderr)
            continue
        if alt.degrees > horizon_deg:
            visible.append((sat, float(alt.degrees), float(az.degrees)))
    visible.sort(key=lambda entry: entry[1], reverse=True)
    return visible


def current_pass_window(
    satellite: EarthSatellite, site: GeographicPosition, horizon_deg: float = 10.0,
    t0: datetime | None = None, lookahead_hours: float = 6.0,
) -> PassWindow:
    """The REMAINING portion of a pass already in progress -- for a
    satellite already confirmed above horizon_deg at t0 (see
    currently_visible_satellites), not a full rise-to-set window like
    find_passes/find_next_pass. t_rise is just t0 itself (the satellite
    already rose before this function is ever called); t_culminate is
    t0 if the real culmination already happened, or the actual upcoming
    culmination event if there's still one ahead; t_set is the next
    horizon-crossing set event. Returned PassWindow feeds
    compute_trajectory(satellite, site, window.t_rise, window.t_set)
    completely unchanged -- "start tracking now" needs no new tracking
    code, just a PassWindow whose rise is "now" instead of a future
    scheduled rise. Raises ValueError if no set event is found within
    lookahead_hours (shouldn't happen for a satellite genuinely above
    horizon_deg, but a pathological orbit/horizon combination could)."""
    ts = load.timescale()
    t0 = t0 or datetime.now(timezone.utc)
    t1 = t0 + timedelta(hours=lookahead_hours)
    times, events = satellite.find_events(site, ts.from_datetime(t0), ts.from_datetime(t1), altitude_degrees=horizon_deg)
    diff = satellite - site

    t_culminate = t0
    t_set = None
    for t, event in zip(times, events):
        if event == 1 and t_set is None:  # culminate, before any set seen yet
            t_culminate = t.utc_datetime()
        elif event == 2:  # set
            t_set = t.utc_datetime()
            break
    if t_set is None:
        raise ValueError(f"no set event found within {lookahead_hours}h -- satellite may not actually be above {horizon_deg} deg")

    t_culm_sf = ts.from_datetime(t_culminate)
    pos_at_culm = diff.at(t_culm_sf)
    alt, _, _ = pos_at_culm.altaz()
    distance_km = float(pos_at_culm.distance().km)
    return PassWindow(
        t_rise=t0,
        t_culminate=t_culminate,
        t_set=t_set,
        max_elevation_deg=float(alt.degrees),
        distance_km=distance_km,
        magnitude_estimate=float("nan"),  # see find_passes' own docstring -- no calibrated magnitude_ref for an arbitrary satellite
    )


@dataclass(frozen=True)
class PassWindow:
    t_rise: datetime
    t_culminate: datetime
    t_set: datetime
    max_elevation_deg: float
    distance_km: float  # slant range at culmination
    magnitude_estimate: float  # see am5.optics.estimate_iss_magnitude -- NaN if magnitude_ref was None (see find_passes)


def find_passes(
    satellite: EarthSatellite,
    site: GeographicPosition,
    t0: datetime | None = None,
    horizon_deg: float = 10.0,
    lookahead_hours: float = 48.0,
    magnitude_ref: float | None = -1.8,
) -> list[PassWindow]:
    """All complete rise/culminate/set passes above `horizon_deg` within the
    lookahead window, in chronological order. Empty list (not an error) if
    none are found — callers that need "the next pass" should raise their
    own error on an empty result, see `find_next_pass`.

    `magnitude_ref` feeds am5.optics.estimate_iss_magnitude's distance
    scaling -- its default is specifically calibrated for the ISS's real
    reflective area, so it's only meaningful for that satellite. Pass None
    (any other satellite's default, see am5/gui/panels.py's PassesPanel)
    to get NaN instead of a fabricated number for an object we have no
    real brightness data for."""
    ts = load.timescale()
    t0 = t0 or datetime.now(timezone.utc)
    t1 = t0 + timedelta(hours=lookahead_hours)
    times, events = satellite.find_events(site, ts.from_datetime(t0), ts.from_datetime(t1), altitude_degrees=horizon_deg)
    diff = satellite - site
    passes = []
    i = 0
    while i < len(events) - 2:
        if events[i] == 0 and events[i + 1] == 1 and events[i + 2] == 2:
            t_rise, t_culm, t_set = times[i], times[i + 1], times[i + 2]
            pos_at_culm = diff.at(t_culm)
            alt, _, _ = pos_at_culm.altaz()
            distance_km = float(pos_at_culm.distance().km)
            passes.append(PassWindow(
                t_rise=t_rise.utc_datetime(),
                t_culminate=t_culm.utc_datetime(),
                t_set=t_set.utc_datetime(),
                max_elevation_deg=float(alt.degrees),
                distance_km=distance_km,
                magnitude_estimate=(
                    estimate_iss_magnitude(distance_km, ref_magnitude=magnitude_ref)
                    if magnitude_ref is not None else float("nan")
                ),
            ))
            i += 3
        else:
            i += 1
    return passes


def find_next_pass(
    satellite: EarthSatellite,
    site: GeographicPosition,
    t0: datetime | None = None,
    horizon_deg: float = 10.0,
    lookahead_hours: float = 24.0,
    magnitude_ref: float | None = -1.8,
) -> PassWindow:
    """First complete rise/culminate/set pass above `horizon_deg` within the
    lookahead window. Raises ValueError if none is found (e.g. horizon too
    high, or a pass is cut off at the edge of the window)."""
    passes = find_passes(satellite, site, t0=t0, horizon_deg=horizon_deg, lookahead_hours=lookahead_hours, magnitude_ref=magnitude_ref)
    if not passes:
        raise ValueError(f"no complete pass above {horizon_deg} deg within {lookahead_hours}h")
    return passes[0]


@dataclass
class Trajectory:
    t_unix: np.ndarray
    ra_deg: np.ndarray  # unwrapped (continuous), may fall outside [0, 360)
    dec_deg: np.ndarray
    dra_dt_deg_s: np.ndarray
    ddec_dt_deg_s: np.ndarray
    alt_deg: np.ndarray
    az_deg: np.ndarray
    ha_hours: np.ndarray  # hour angle, for meridian_crossings — sign flips at the meridian
    distance_km: np.ndarray  # slant range, for am5/optics.py's magnitude/exposure estimates

    def sky_speed_deg_s(self) -> np.ndarray:
        """On-sky angular speed (not raw dRA/dt -- see am5/optics.py)."""
        return np.sqrt((self.dra_dt_deg_s * np.cos(np.radians(self.dec_deg))) ** 2 + self.ddec_dt_deg_s ** 2)

    def distance_at(self, t_unix: float) -> float:
        return float(np.interp(t_unix, self.t_unix, self.distance_km))

    def interpolate(self, t_unix: float) -> tuple[float, float, float, float]:
        """(ra_deg, dec_deg, dra_dt_deg_s, ddec_dt_deg_s) at `t_unix`, linearly
        interpolated. `ra_deg` may be outside [0, 360) since the stored series
        is unwrapped — reduce mod 360 before sending it over the wire.

        Position clamps to the trajectory's first/last sample outside its
        own time range (np.interp's default behaviour) -- fine, that's just
        "where the pass starts/ends". But the RATE at that boundary is a
        real, usually large, angular velocity -- extrapolating it forever
        outside the window used to mean starting the tracking loop early
        made the mount continuously slew away at that rate instead of
        holding still while waiting for the pass to actually begin. Rates
        are explicitly zeroed outside [t_unix[0], t_unix[-1]] so an early
        start just sits at the boundary position until real time enters the
        window, then tracks normally."""
        ra = float(np.interp(t_unix, self.t_unix, self.ra_deg))
        dec = float(np.interp(t_unix, self.t_unix, self.dec_deg))
        if self.t_unix[0] <= t_unix <= self.t_unix[-1]:
            dra = float(np.interp(t_unix, self.t_unix, self.dra_dt_deg_s))
            ddec = float(np.interp(t_unix, self.t_unix, self.ddec_dt_deg_s))
        else:
            dra = ddec = 0.0
        return ra, dec, dra, ddec


def compute_trajectory(
    satellite: EarthSatellite,
    site: GeographicPosition,
    t_start: datetime,
    t_end: datetime,
    step_s: float = 0.05,
) -> Trajectory:
    """Precompute the whole pass at a fixed timestep, apparent-of-date RA/DEC
    (what :SMeq#/:GMEQ# expect — not J2000). Rates come from a numerical
    central difference over the sampled series rather than an analytic
    formula: simpler, and the 20Hz sampling is already far finer than the
    ISS's angular acceleration needs."""
    if t_end <= t_start:
        raise ValueError("t_end must be after t_start")
    ts = load.timescale()
    duration_s = (t_end - t_start).total_seconds()
    offsets_s = np.arange(0.0, duration_s, step_s)
    t_unix = t_start.timestamp() + offsets_s
    tarr = ts.utc(
        t_start.year, t_start.month, t_start.day, t_start.hour, t_start.minute,
        t_start.second + t_start.microsecond / 1e6 + offsets_s,
    )

    diff = satellite - site
    pos = diff.at(tarr)
    ra, dec, _ = pos.radec(epoch="date")
    alt, az, _ = pos.altaz()
    ha, _, _ = pos.hadec()

    ra_deg = unwrap_deg(ra.degrees)
    dec_deg = dec.degrees
    dra_dt = np.gradient(ra_deg, t_unix)
    ddec_dt = np.gradient(dec_deg, t_unix)

    return Trajectory(
        t_unix=t_unix,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        dra_dt_deg_s=dra_dt,
        ddec_dt_deg_s=ddec_dt,
        alt_deg=alt.degrees,
        az_deg=az.degrees,
        ha_hours=ha.hours,
        distance_km=pos.distance().km,
    )


def meridian_crossings(trajectory: Trajectory) -> list[datetime]:
    """UTC timestamps where the hour angle changes sign — where a German
    equatorial mount would need a meridian flip to keep tracking. Purely
    informational: nothing in this codebase acts on this automatically, the
    operator picks the starting pier side by hand."""
    sign_changes = np.where(np.diff(np.sign(trajectory.ha_hours)) != 0)[0]
    return [datetime.fromtimestamp(float(trajectory.t_unix[i]), tz=timezone.utc) for i in sign_changes]
