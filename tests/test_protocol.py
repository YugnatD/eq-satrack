
import pytest

from am5 import protocol


def test_parse_ra_hours():
    assert protocol.parse_ra_hours("12:30:00") == pytest.approx(12.5)
    assert protocol.parse_ra_hours("00:00:00") == 0.0
    assert protocol.parse_ra_hours("23:59:59.99") == pytest.approx(24.0, abs=1e-4)


def test_format_ra_hours_roundtrip():
    for h in (0.0, 6.25, 12.5, 23.999):
        s = protocol.format_ra_hours(h)
        assert protocol.parse_ra_hours(s) == pytest.approx(h % 24.0, abs=1e-3)


def test_parse_signed_dms():
    assert protocol.parse_signed_dms("+45*30:00") == pytest.approx(45.5)
    assert protocol.parse_signed_dms("-45*30:00") == pytest.approx(-45.5)
    assert protocol.parse_signed_dms("+00*00:00") == 0.0


def test_format_signed_dms_roundtrip():
    for d in (-89.5, -0.25, 0.0, 45.75, 89.99):
        s = protocol.format_signed_dms(d)
        assert protocol.parse_signed_dms(s) == pytest.approx(d, abs=1e-2)


def test_format_signed_dms_sign_and_padding():
    assert protocol.format_signed_dms(5.0) == "+05*00:00"
    assert protocol.format_signed_dms(-5.0) == "-05*00:00"


def test_format_dms_never_emits_60_seconds_or_minutes():
    # regression: naive per-field rounding produced e.g. "-88*56:60"
    # (invalid sexagesimal -> e2# on real hardware, silent GOTO failure).
    for deg in (-88.95, 45.9931, -10.4986, 12.9958, 89.99986, -0.00001):
        s = protocol.format_signed_dms(deg)
        minutes = int(s.split("*")[1].split(":")[0])
        seconds = int(s.split(":")[1])
        assert 0 <= minutes <= 59, s
        assert 0 <= seconds <= 59, s
        assert protocol.parse_signed_dms(s) == pytest.approx(deg, abs=0.6 / 3600)
    for az in (359.9999, 179.99986, 5.9931):
        s = protocol.format_unsigned_dms(az)
        assert 0 <= int(s.split("*")[1].split(":")[0]) <= 59, s
        assert 0 <= int(s.split(":")[1]) <= 59, s


def test_format_ra_hours_never_emits_60_seconds():
    for h in (23.99999, 12.99986, 5.51663):
        s = protocol.format_ra_hours(h)
        assert 0 <= int(s.split(":")[2].split(".")[0]) <= 59, s
        assert 0 <= int(s.split(":")[1]) <= 59, s


def test_build_sd_stays_valid_at_problem_declination():
    assert protocol.build_sd(-88.95) == b":Sd-88*57:00#"


def test_parse_unsigned_dms():
    assert protocol.parse_unsigned_dms("180*00:00") == pytest.approx(180.0)
    assert protocol.parse_unsigned_dms("000*00:00") == 0.0


def test_format_unsigned_dms_roundtrip():
    for d in (0.0, 90.25, 180.0, 359.99):
        s = protocol.format_unsigned_dms(d)
        assert protocol.parse_unsigned_dms(s) == pytest.approx(d, abs=1e-2)


def test_parse_error():
    assert protocol.parse_error("e2#") == 2
    assert protocol.parse_error("e12#") == 12
    assert protocol.parse_error("1#") is None
    assert protocol.parse_error("N/A#") is None


def test_parse_geq_ok():
    radec = protocol.parse_geq("12:00:00&+45*30:00#")
    assert radec.ra_hours == pytest.approx(12.0)
    assert radec.dec_deg == pytest.approx(45.5)


def test_parse_geq_negative_dec():
    radec = protocol.parse_geq("06:15:30&-10*15:30#")
    assert radec.ra_hours == pytest.approx(6.25833, abs=1e-4)
    assert radec.dec_deg == pytest.approx(-10.25833, abs=1e-4)


def test_parse_geq_error_reply_raises():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_geq("e7#")


def test_parse_geq_malformed_raises():
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_geq("garbage#")


def test_parse_gza_ok():
    azalt = protocol.parse_gza("180*00:00&+45*00:00#")
    assert azalt.az_deg == pytest.approx(180.0)
    assert azalt.alt_deg == pytest.approx(45.0)


def test_build_rv():
    assert protocol.build_rv(239.0) == b":Rv239.00#"
    assert protocol.build_rv(0.0) == b":Rv0.00#"
    assert protocol.build_rv(1440.0) == b":Rv1440.00#"


def test_build_rv_out_of_range():
    with pytest.raises(ValueError):
        protocol.build_rv(1440.01)
    with pytest.raises(ValueError):
        protocol.build_rv(-0.01)


def test_build_move_and_quit():
    assert protocol.build_move("e") == b":Me#"
    assert protocol.build_quit("n") == b":Qn#"
    assert protocol.build_quit(None) == b":Q#"
    with pytest.raises(ValueError):
        protocol.build_move("x")


def test_format_ra_hours_int():
    assert protocol.format_ra_hours_int(12.5) == "12:30:00"
    assert protocol.format_ra_hours_int(0.0) == "00:00:00"
    assert protocol.format_ra_hours_int(23.999999999) == "00:00:00"  # rounds up into the next day, wraps


def test_build_sr_and_sd():
    assert protocol.build_sr(12.5) == b":Sr12:30:00#"
    assert protocol.build_sd(45.5) == b":Sd+45*30:00#"
    assert protocol.build_sd(-10.0) == b":Sd-10*00:00#"


def test_build_slew_and_sync():
    assert protocol.build_slew() == b":MS#"
    assert protocol.build_sync() == b":CM#"


def test_build_park():
    assert protocol.build_park() == b":hP#"


def test_build_set_timezone_sign_is_inverted():
    # doc's own example: UTC+8 (China) is sent as "-08:00", not "+08:00"
    assert protocol.build_set_timezone(8.0) == b":SG-08:00#"
    assert protocol.build_set_timezone(-5.0) == b":SG+05:00#"
    assert protocol.build_set_timezone(0.0) == b":SG+00:00#"
    assert protocol.build_set_timezone(5.5) == b":SG-05:30#"  # e.g. India, fractional-hour zone


def test_build_set_latitude_and_longitude():
    assert protocol.build_set_latitude(46.18) == b":St+46*10:48#"
    assert protocol.build_set_latitude(-33.5) == b":St-33*30:00#"
    assert protocol.build_set_longitude(6.14) == b":Sg+006*08:24#"
    assert protocol.build_set_longitude(-118.25) == b":Sg-118*15:00#"


def test_build_set_date_time_timezone():
    from datetime import datetime, timedelta, timezone

    when = datetime(2026, 7, 10, 20, 0, 0, tzinfo=timezone(timedelta(hours=1)))
    assert protocol.build_set_date_time_timezone(when) == b":SMTI07/10/26&20:00:00&-01:00#"


def test_parse_gu_status_flags():
    status = protocol.parse_gu_status("nNG001000060#")
    assert status.is_parked is False
    assert status.is_equatorial is True
    assert status.is_at_home is False

    status = protocol.parse_gu_status("NHZP001000060#")
    assert status.is_parked is True
    assert status.is_equatorial is False
    assert status.is_at_home is True


def test_parse_gu_status_stall_flags_case_sensitive():
    status = protocol.parse_gu_status("NG S s001000060#")
    assert status.ra_stalled is True
    assert status.dec_stalled is True

    status = protocol.parse_gu_status("NG001000060#")
    assert status.ra_stalled is False
    assert status.dec_stalled is False


def test_parse_gu_status_no_mode_char_is_none():
    status = protocol.parse_gu_status("nN001000060#")
    assert status.is_equatorial is None
