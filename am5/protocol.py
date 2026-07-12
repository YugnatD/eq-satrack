"""LX200-derived wire format used by the ZWO AM5.

All commands are ASCII, start with ':' and end with '#'. Two sexagesimal
flavours are used: unsigned "DDD*MM:SS" (azimuth) and signed "sDD*MM:SS"
(declination, altitude), where '*' is the literal degree separator, not a
placeholder. RA uses plain "HH:MM:SS". Compound responses (:GMEQ#, :GMZA#)
join their two sexagesimal fields with '&'.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


class ProtocolError(ValueError):
    """Raised when a mount reply cannot be parsed or is an error code."""


def _split_dms(body: str) -> tuple[int, int, float]:
    # body: "DD*MM:SS" or "DDD*MM:SS" (no sign)
    deg_str, rest = body.split("*", 1)
    min_str, sec_str = rest.split(":", 1)
    return int(deg_str), int(min_str), float(sec_str)


def parse_ra_hours(field: str) -> float:
    """Parse 'HH:MM:SS' into hours (0-24)."""
    h_str, m_str, s_str = field.split(":", 2)
    h, m, s = int(h_str), int(m_str), float(s_str)
    return h + m / 60.0 + s / 3600.0


def format_ra_hours(hours: float) -> str:
    # Round to whole centiseconds first, then carry-decompose -- computing
    # h/m/s independently and formatting seconds with %05.2f rounds e.g.
    # 59.996s up to "60.00", producing an invalid "HH:MM:60.00" string.
    total_cs = round((hours % 24.0) * 3600.0 * 100.0) % (24 * 3600 * 100)
    total_s, cs = divmod(total_cs, 100)
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"


def format_ra_hours_int(hours: float) -> str:
    """'HH:MM:SS' with integer seconds — the :Sr# target-RA command takes no
    fractional seconds, unlike the :GMEQ# reply format above."""
    total_seconds = round((hours % 24.0) * 3600.0) % 86400
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_signed_dms(field: str) -> float:
    """Parse 'sDD*MM:SS' (declination/altitude) into signed degrees."""
    sign = -1.0 if field[0] == "-" else 1.0
    body = field[1:] if field[0] in "+-" else field
    d, m, s = _split_dms(body)
    return sign * (d + m / 60.0 + s / 3600.0)


def format_signed_dms(degrees: float, deg_width: int = 2) -> str:
    # Round to whole arcseconds then carry-decompose. The naive
    # per-field version (int(deg), int(min), then "%02.0f" on seconds)
    # rounded e.g. 30'59.7" up to ":60", producing an out-of-range
    # "sDD*MM:60" that real hardware rejects with e2# -- confirmed to hit
    # ~1% of DEC targets, silently failing GOTO/sync at those declinations.
    sign = "-" if degrees < 0 else "+"
    total_arcsec = round(abs(degrees) * 3600.0)
    d, rem = divmod(total_arcsec, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{d:0{deg_width}d}*{m:02d}:{s:02d}"


def parse_unsigned_dms(field: str) -> float:
    """Parse 'DDD*MM:SS' (azimuth) into degrees (0-360)."""
    d, m, s = _split_dms(field)
    return d + m / 60.0 + s / 3600.0


def format_unsigned_dms(degrees: float, deg_width: int = 3) -> str:
    # Carry-decompose from rounded arcseconds -- see format_signed_dms.
    total_arcsec = round((degrees % 360.0) * 3600.0) % (360 * 3600)
    d, rem = divmod(total_arcsec, 3600)
    m, s = divmod(rem, 60)
    return f"{d:0{deg_width}d}*{m:02d}:{s:02d}"


def _strip_frame(raw: str) -> str:
    raw = raw.strip()
    if raw.endswith("#"):
        raw = raw[:-1]
    return raw


def parse_error(raw: str) -> int | None:
    """Return the error code if `raw` is an 'e<code>#' reply, else None."""
    body = _strip_frame(raw)
    if len(body) >= 2 and body[0] == "e" and body[1:].isdigit():
        return int(body[1:])
    return None


@dataclass(frozen=True)
class RaDec:
    ra_hours: float
    dec_deg: float


@dataclass(frozen=True)
class AzAlt:
    az_deg: float
    alt_deg: float


def parse_geq(raw: str) -> RaDec:
    """Parse a :GMEQ# reply: 'HH:MM:SS&sDD*MM:SS#'."""
    body = _strip_frame(raw)
    err = parse_error(raw)
    if err is not None:
        raise ProtocolError(f"mount returned error e{err}")
    try:
        ra_field, dec_field = body.split("&", 1)
        return RaDec(parse_ra_hours(ra_field), parse_signed_dms(dec_field))
    except ValueError as exc:
        raise ProtocolError(f"malformed :GMEQ# reply {raw!r}") from exc


def parse_gza(raw: str) -> AzAlt:
    """Parse a :GMZA# reply: 'DDD*MM:SS&sDD*MM:SS#'."""
    body = _strip_frame(raw)
    err = parse_error(raw)
    if err is not None:
        raise ProtocolError(f"mount returned error e{err}")
    try:
        az_field, alt_field = body.split("&", 1)
        return AzAlt(parse_unsigned_dms(az_field), parse_signed_dms(alt_field))
    except ValueError as exc:
        raise ProtocolError(f"malformed :GMZA# reply {raw!r}") from exc


def build_rv(rate_x_sidereal: float) -> bytes:
    """':Rv#' variable slew rate, 0.00-1440.00 x sidereal, no zero-padding."""
    if not 0.0 <= rate_x_sidereal <= 1440.0:
        raise ValueError(f"rate {rate_x_sidereal} out of range [0, 1440]")
    return f":Rv{rate_x_sidereal:.2f}#".encode("ascii")


def build_move(direction: str) -> bytes:
    if direction not in "ewns":
        raise ValueError(f"invalid direction {direction!r}, expected one of e/w/n/s")
    return f":M{direction}#".encode("ascii")


def build_quit(direction: str | None = None) -> bytes:
    if direction is None:
        return b":Q#"
    if direction not in "ewns":
        raise ValueError(f"invalid direction {direction!r}, expected one of e/w/n/s")
    return f":Q{direction}#".encode("ascii")


def build_go_home() -> bytes:
    """:hC# — slew to the mount's home/zero position. No reply, per the
    brief's own protocol table."""
    return b":hC#"


def build_park() -> bytes:
    """:hP# — ZWO's native park (untested against real hardware — INDI's own
    driver avoids it, see Mount.park()'s docstring). Replies '1#'/'0#',
    unlike :hC#."""
    return b":hP#"


# The brief's original "SMeq#/SMMC#" table entries do not exist on the wire
# (confirmed against real hardware: e2# format-error regardless of
# separator). The real LX200-family target-setting sequence, per upstream
# indilib/indi's lx200driver.cpp (which lx200am5.cpp inherits verbatim,
# it defines no Goto/Sync of its own) is three separate commands. Unlike
# the rest of the protocol, :Sr#/:Sd# reply with a single raw '1'/'0'
# character and no '#' terminator — see Transport.read_exact.


def build_sr(ra_hours: float) -> bytes:
    """:SrHH:MM:SS# — stage the RA half of a GOTO/sync target."""
    return f":Sr{format_ra_hours_int(ra_hours)}#".encode("ascii")


def build_sd(dec_deg: float) -> bytes:
    """:SdsDD*MM:SS# — stage the DEC half of a GOTO/sync target."""
    return f":Sd{format_signed_dms(dec_deg)}#".encode("ascii")


def build_slew() -> bytes:
    """:MS# — slew to the previously staged :Sr#/:Sd# target."""
    return b":MS#"


def build_sync() -> bytes:
    """:CM# — sync to the previously staged :Sr#/:Sd# target, no motion."""
    return b":CM#"


# The commands and :GU# field meanings below come from ZWO's own protocol
# PDF (docs/ZWO Mount Serial Communication Protocol_v1.7.pdf), not a
# transcription or a third-party driver — read directly with the Read tool.
# It also confirms the :Sr#/:Sd#/:MS# sequence above matches its own
# "Procedure example" flow diagram exactly.


def build_set_timezone(utc_offset_hours: float) -> bytes:
    """:SGsHH:MM# — set time zone. The wire value is NEGATED from the usual
    UTC+N convention: the doc's own example sets UTC+8 (China) as "SG-08#",
    not "+08" — the same inverted-sign convention POSIX TZ strings use."""
    wire_offset_minutes = round(-utc_offset_hours * 60.0)
    sign = "+" if wire_offset_minutes >= 0 else "-"
    hh, mm = divmod(abs(wire_offset_minutes), 60)
    return f":SG{sign}{hh:02d}:{mm:02d}#".encode("ascii")


def build_set_latitude(latitude_deg: float) -> bytes:
    """:StsDD*MM:SS# — set observer latitude."""
    return f":St{format_signed_dms(latitude_deg, deg_width=2)}#".encode("ascii")


def build_set_longitude(longitude_deg: float) -> bytes:
    """:SgsDDD*MM:SS# — set observer longitude (3-digit degrees, unlike
    latitude's 2)."""
    return f":Sg{format_signed_dms(longitude_deg, deg_width=3)}#".encode("ascii")


def build_set_date_time_timezone(when: datetime) -> bytes:
    """:SMTIMM/DD/YY&HH:MM:SS&sHH:MM# — compound command setting date, time
    and time zone in one shot (from the doc's own initialization flow
    diagram). `when` should be timezone-aware, in the zone you want the
    mount configured for; its own utcoffset() is used, sign-inverted per
    build_set_timezone()'s note. Pass e.g. datetime.now().astimezone() for
    "whatever this computer's local time zone currently is"."""
    offset = when.utcoffset()
    offset_minutes = round(offset.total_seconds() / 60.0) if offset else 0
    wire_offset_minutes = -offset_minutes
    sign = "+" if wire_offset_minutes >= 0 else "-"
    hh, mm = divmod(abs(wire_offset_minutes), 60)
    date_str = when.strftime("%m/%d/%y")
    time_str = when.strftime("%H:%M:%S")
    return f":SMTI{date_str}&{time_str}&{sign}{hh:02d}:{mm:02d}#".encode("ascii")


@dataclass(frozen=True)
class MountStatus:
    """Best-effort parse of a :GU# reply. The doc lists which characters
    can appear (each individually documented as "or not shown", i.e. the
    string is variable-length depending on which conditions are active) but
    never gives a fixed field layout — so rather than guess at character
    positions, this checks for presence of each flag character. Safe: the
    letter-flags (n/N/L/H/G/Z/S/s/T/t/P) never collide with the numeric
    rate/state fields also present in the string."""

    raw: str
    is_parked: bool
    is_equatorial: bool | None  # None if neither G nor Z was found
    is_at_home: bool
    ra_stalled: bool
    dec_stalled: bool


def parse_gu_status(raw: str) -> MountStatus:
    body = _strip_frame(raw)
    return MountStatus(
        raw=raw,
        is_parked="P" in body,
        is_equatorial=True if "G" in body else (False if "Z" in body else None),
        is_at_home="H" in body,
        ra_stalled="S" in body,
        dec_stalled="s" in body,
    )
