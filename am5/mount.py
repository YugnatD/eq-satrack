"""High-level, typed wrapper around a `Transport` implementing the AM5 command set."""

from __future__ import annotations

from datetime import datetime

from . import protocol
from .protocol import MountStatus, ProtocolError, parse_error
from .transport import Transport


class Mount:
    def __init__(self, transport: Transport, response_timeout: float = 2.0):
        self._t = transport
        self._timeout = response_timeout

    def _send(self, cmd: bytes, expect_response: bool, timeout: float | None = None) -> str:
        self._t.write(cmd)
        if not expect_response:
            return ""
        raw = self._t.read_until_hash(timeout if timeout is not None else self._timeout)
        return raw.decode("ascii", errors="replace")

    def _send_single_char(self, cmd: bytes, timeout: float | None = None) -> str:
        self._t.write(cmd)
        raw = self._t.read_exact(1, timeout if timeout is not None else self._timeout)
        return raw.decode("ascii", errors="replace")

    def get_radec(self) -> protocol.RaDec:
        return protocol.parse_geq(self._send(b":GMEQ#", expect_response=True))

    def get_azalt(self) -> protocol.AzAlt:
        return protocol.parse_gza(self._send(b":GMZA#", expect_response=True))

    def set_rate(self, rate_x_sidereal: float) -> None:
        self._send(protocol.build_rv(rate_x_sidereal), expect_response=False)

    def move(self, direction: str) -> None:
        self._send(protocol.build_move(direction), expect_response=False)

    def stop(self, direction: str | None = None) -> None:
        self._send(protocol.build_quit(direction), expect_response=False)

    def park(self) -> None:
        """:hC# — slew to the mount's home/zero position. Used as "park"
        here rather than the native :hP#: INDI's own lx200am5 driver
        deliberately does the same (goHome() instead of a real park
        command), because ZWO's native park position is horizontal, not
        the conventional counterweight-down pointed-at-pole position this
        project's meridian planning assumes.

        Confirmed (via INDI's own source, not a guess) that there is no
        wire-level "unpark" for THIS path: :hC# is just a GOTO to a
        reference position, not a locked hardware state, so a parked mount
        just needs a fresh motion command — the "parked" concept only
        exists as a local safety flag (see am5/gui/worker.py), matching
        INDI's UnPark() which touches no hardware either.

        This does NOT necessarily hold for park_native() below — untested."""
        self._send(protocol.build_go_home(), expect_response=False)

    def park_native(self) -> str:
        """:hP# — ZWO's native park. Per the official protocol doc, this
        only works in equatorial mode — callers should check
        get_status().is_equatorial first (Mount itself does no client-side
        validation, same as everywhere else in this class). Whether this
        leaves the mount in a real locked hardware state that needs an
        explicit wire-level unpark command is UNKNOWN — INDI's own driver
        avoids this path entirely (see park()'s docstring), so there is no
        reference implementation to confirm either way. Returns the raw
        '1#'/'0#' reply."""
        return self._send(protocol.build_park(), expect_response=True)

    def emergency_stop(self) -> None:
        """Best-effort :Q# — never raises, safe to call from a signal handler or `finally`."""
        try:
            self._t.write(b":Q#")
        except Exception:
            pass

    def set_tracking(self, on: bool) -> str:
        # :Te#/:Td# reply "1"/"0" with no '#' terminator (confirmed against
        # the protocol doc and real hardware -- _send()'s read_until_hash
        # blocked for the full 2s response_timeout on every single call
        # before this fix, since that '#' never arrives).
        return self._send_single_char(b":Te#" if on else b":Td#")

    def get_tracking_status(self) -> str:
        return self._send(b":GAT#", expect_response=True)

    def get_status_raw(self) -> str:
        return self._send(b":GU#", expect_response=True)

    def get_status(self) -> MountStatus:
        return protocol.parse_gu_status(self.get_status_raw())

    def sync_site_and_time(self, latitude_deg: float, longitude_deg: float, when: datetime | None = None) -> None:
        """Runs the site/time half of the doc's own "Procedure example" init
        sequence (:SMTI# then :St#/:Sg# — not :SC#/:SL#/:SG# separately,
        the compound command covers those). Without this, :MS# rejects
        every GOTO with e7# ("time/location not synced") — confirmed
        against real hardware. `when` defaults to this computer's local
        time (datetime.now().astimezone()), assuming chronyd/NTP keeps it
        accurate per the brief's own requirement — pass an explicit value
        to override.

        All three replies are "1"/"0" with no '#' terminator (per the
        protocol doc's "Response: 1: Success, 0: False" for :SMTI#/:St#/
        :Sg#) -- confirmed on real hardware this was blocking 2s *per
        command* (6s total, on every single connect) before switching to
        _send_single_char."""
        when = when or datetime.now().astimezone()
        for label, cmd in (
            ("date/time/timezone", protocol.build_set_date_time_timezone(when)),
            ("latitude", protocol.build_set_latitude(latitude_deg)),
            ("longitude", protocol.build_set_longitude(longitude_deg)),
        ):
            reply = self._send_single_char(cmd).strip()
            if reply not in ("1", "1#"):
                raise ProtocolError(f"mount rejected {label}: {reply!r}")

    def get_version(self) -> str:
        return self._send(b":GV#", expect_response=True)

    def get_pier_side(self) -> str:
        """:Gm# -- current cardinal orientation of the mount: 'E', 'W', or
        'N' (home/zero position, no direction). No wire command exists to
        set this directly (checked against the official protocol doc) --
        the firmware alone decides which side a GOTO lands on, based on the
        target's hour angle at the moment the command is sent. This is
        read-only, for the operator to check before committing to a
        tracking run."""
        return self._send(b":Gm#", expect_response=True).strip().rstrip("#")

    def set_meridian_behavior(self, track_past_meridian: bool, limit_deg: float, flip: bool = False) -> None:
        """:ST<nnsnn># -- see protocol.build_set_meridian_behavior's
        docstring for why this exists (root cause of a real "diverges
        badly after the meridian" incident: never configured before,
        default mount behavior silently stops RA tracking ~1 deg past the
        meridian). Reply format is UNCONFIRMED against real hardware --
        the doc's own "1: Success, 0: False" wording is identical to
        set_tracking's/set_altitude_limits_enabled's, both of which turned
        out to be a bare, unterminated "1"/"0" only after a real-hardware
        test caught _send() blocking for the full response_timeout
        expecting a '#' that never came, so this uses the same
        _send_single_char path defensively -- treat as unverified until
        actually tested."""
        reply = self._send_single_char(protocol.build_set_meridian_behavior(track_past_meridian, limit_deg, flip)).strip()
        if reply not in ("1", "1#"):
            raise ProtocolError(f"mount rejected meridian behavior: {reply!r}")

    def get_meridian_behavior(self) -> tuple[bool, bool, float]:
        """:GTa# -- (flip, track_past_meridian, limit_deg), see
        protocol.parse_meridian_behavior. Doc shows this reply IS
        '#'-terminated ('nnsnn#'), unlike the Set command above."""
        return protocol.parse_meridian_behavior(self._send(b":GTa#", expect_response=True))

    def set_altitude_limits_enabled(self, enabled: bool) -> str:
        # :SLE#/:SLD# reply "1"/"0" with no '#' terminator -- same class of
        # bug as set_tracking() had (confirmed on real hardware: this was
        # blocking for the full 2s response_timeout on every single call
        # before this fix, since read_until_hash's '#' never arrives).
        return self._send_single_char(b":SLE#" if enabled else b":SLD#")

    def set_target_ra(self, ra_hours: float) -> bool:
        """:Sr# — stage the RA half of a GOTO/sync target. Single raw
        character reply ('1' accepted / '0' rejected), no '#' terminator —
        unlike the rest of the protocol (see protocol.py)."""
        return self._send_single_char(protocol.build_sr(ra_hours)) == "1"

    def set_target_dec(self, dec_deg: float) -> bool:
        """:Sd# — stage the DEC half of a GOTO/sync target."""
        return self._send_single_char(protocol.build_sd(dec_deg)) == "1"

    def slew_to_target(self) -> int:
        """:MS# — slew to the previously staged :Sr#/:Sd# target. Confirmed
        against real hardware to be '#'-terminated like the rest of the
        protocol (not the bare single-digit, no-terminator convention of the
        generic LX200 driver this firmware otherwise resembles) — real
        AM3 replied 'e7#' (time/location not synced), not a lone '7'.
        Returns 0 slewing / 1 below horizon / 2 below the altitude limit, or
        the negation of an e<code># error code (e.g. -7 for e7#).
        Completion is not signalled by this reply — poll get_radec() until
        position stops changing (see characterize.py test h)."""
        raw = self._send(protocol.build_slew(), expect_response=True).strip()
        err = parse_error(raw)
        if err is not None:
            return -err
        digits = raw.rstrip("#")
        return int(digits) if digits.isdigit() else -1

    def sync_to_target(self) -> str:
        """:CM# — sync to the previously staged :Sr#/:Sd# target, no motion."""
        return self._send(protocol.build_sync(), expect_response=True)

    def goto(self, ra_hours: float, dec_deg: float) -> int:
        """:Sr#+:Sd#+:MS# — stage a target and slew to it. Raises
        ProtocolError if the mount rejects the staged target outright;
        otherwise returns :MS#'s result digit (see slew_to_target)."""
        if not self.set_target_ra(ra_hours):
            raise ProtocolError(f"mount rejected RA target {ra_hours}h")
        if not self.set_target_dec(dec_deg):
            raise ProtocolError(f"mount rejected DEC target {dec_deg} deg")
        return self.slew_to_target()

    def sync(self, ra_hours: float, dec_deg: float) -> str:
        """:Sr#+:Sd#+:CM# — stage a target and sync to it without moving."""
        if not self.set_target_ra(ra_hours):
            raise ProtocolError(f"mount rejected RA target {ra_hours}h")
        if not self.set_target_dec(dec_deg):
            raise ProtocolError(f"mount rejected DEC target {dec_deg} deg")
        return self.sync_to_target()

    def close(self) -> None:
        self._t.close()
