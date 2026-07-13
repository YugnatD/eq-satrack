import time
import tkinter as tk
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from skyfield.api import EarthSatellite, load, wgs84

from am5.clock_sync import ClockSyncStatus
from am5.ephemeris import PassWindow, Trajectory, compute_trajectory, find_next_pass, meridian_crossings
from am5.gui.panels import (
    CUSTOM_SATELLITE_LABEL,
    GUIDING_CALIB_NUDGE_DURATION_S,
    GUIDING_CALIB_NUDGE_RATE_X,
    GUIDING_PERP_PULSE_DURATION_S,
    KNOWN_SATELLITES,
    MAX_TRACKING_DURATION_S,
    ConnectionPanel,
    ExposurePanel,
    CalibrationPanel,
    PassesPanel,
    SiteVars,
    TransitPanel,
    _local_and_utc,
    _meridian_detail_line,
    _sanitize_filename,
)
from am5.gui.worker import MountWorker, WorkerEvent
from am5.optics import OpticalTrain
from am5.tracker import LiveOffsets
from camera.guiding import BlobDetection, GuidingCalibration
from camera.worker import CameraEvent, CameraWorker, frame_to_pgm

# Same fixed, network-free TLE as tests/test_ephemeris.py.
_TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9006"
_TLE_LINE2 = "2 25544  51.6400 208.9163 0006317  69.9862 25.2825 15.49560500000000"


def _tk_available() -> bool:
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


pytestmark = pytest.mark.skipif(not _tk_available(), reason="no Tk display available")


class _StubMapWidget(tk.Frame):
    """Stands in for tkintermapview.TkinterMapView in tests -- the real
    widget starts ~26 daemon threads per instance (see ConnectionPanel's
    map_widget_cls comment) that can segfault the interpreter at process
    exit once enough pile up across a test run constructing many
    ConnectionPanels. Implements only what ConnectionPanel actually calls."""

    def __init__(self, parent, width=380, height=280, corner_radius=0, **_kwargs):
        super().__init__(parent, width=width, height=height)

    def set_position(self, *_args, **_kwargs):
        pass

    def set_zoom(self, *_args, **_kwargs):
        pass

    def delete_all_marker(self):
        pass

    def set_marker(self, *_args, **_kwargs):
        pass

    def add_left_click_map_command(self, _callback):
        pass


def _window(t_rise: datetime, duration_s: float = 300.0, max_elevation_deg: float = 45.0, magnitude_estimate: float = -2.0) -> PassWindow:
    return PassWindow(
        t_rise=t_rise, t_culminate=t_rise + timedelta(seconds=duration_s / 2),
        t_set=t_rise + timedelta(seconds=duration_s), max_elevation_deg=max_elevation_deg,
        distance_km=500.0, magnitude_estimate=magnitude_estimate,
    )


def test_local_and_utc_shows_both_and_they_can_differ():
    dt = datetime(2026, 7, 10, 19, 22, 20, tzinfo=timezone.utc)
    text = _local_and_utc(dt)
    assert "19:22:20 UTC" in text
    assert "local" in text


def test_meridian_detail_line_no_crossing():
    w = _window(datetime.now(timezone.utc))
    assert _meridian_detail_line([], w) == "No meridian crossing during this pass"


def test_meridian_detail_line_reports_offsets():
    rise = datetime(2026, 7, 10, 19, 22, 20, tzinfo=timezone.utc)
    w = _window(rise, duration_s=300.0)  # culminate at rise+150s
    crossing = rise + timedelta(seconds=140.0)  # 10s before culmination
    line = _meridian_detail_line([crossing], w)
    assert "MERIDIAN CROSSING" in line
    assert "140s after tracking starts" in line
    assert "10s before culmination" in line


@pytest.fixture
def panel(tmp_path):
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    p = TransitPanel(root, mount_worker, camera_worker, tmp_path)
    yield p
    mount_worker.shutdown()
    camera_worker.shutdown()
    root.destroy()


def test_check_pass_timing_allows_starting_hours_early(panel):
    # Starting early is a deliberate operator choice (arm, start, let it
    # wait) -- only a pass that's already over is refused, see
    # _check_pass_timing's docstring.
    panel._window = _window(datetime.now(timezone.utc) + timedelta(hours=2))
    with patch("am5.gui.panels.messagebox.showerror") as showerror:
        assert panel._check_pass_timing() is True
        showerror.assert_not_called()


def test_check_pass_timing_blocks_after_set(panel):
    panel._window = _window(datetime.now(timezone.utc) - timedelta(seconds=600), duration_s=60.0)
    with patch("am5.gui.panels.messagebox.showerror") as showerror:
        assert panel._check_pass_timing() is False
        showerror.assert_called_once()


def test_check_pass_timing_true_with_no_window_selected(panel):
    panel._window = None
    with patch("am5.gui.panels.messagebox.showerror") as showerror:
        assert panel._check_pass_timing() is True
        showerror.assert_not_called()


def test_on_start_click_caps_duration_even_for_a_far_future_set_time(panel):
    # Simulates the guard having been bypassed (e.g. a stale _window set
    # after the clock moved on): duration_s must never balloon to hours.
    panel._window = _window(datetime.now(timezone.utc), duration_s=10 * 3600.0)
    panel._trajectory = object()  # only needs to be non-None for this codepath
    panel._armed = True
    captured = {}
    panel._mount_worker.start_tracking = lambda *args, **kwargs: captured.setdefault("duration_s", args[4])
    panel._on_start_click()
    assert captured["duration_s"] == pytest.approx(MAX_TRACKING_DURATION_S)


def test_on_start_click_passes_mount_lag_and_feedback_into_tracking_config(panel):
    panel._window = _window(datetime.now(timezone.utc), duration_s=60.0)
    panel._trajectory = object()
    panel._armed = True
    panel._mount_lag_var.set(0.27)
    panel._feedback_enabled_var.set(True)
    captured = {}
    panel._mount_worker.start_tracking = lambda *args, **kwargs: captured.setdefault("config", args[5])
    panel._on_start_click()
    assert captured["config"].mount_lag_s == pytest.approx(0.27)
    assert captured["config"].enable_feedback is True


def test_build_tracking_config_falls_back_to_zero_lag_on_invalid_input(panel):
    panel._mount_lag_var = tk.StringVar(value="not-a-number")  # simulate a bad manual edit
    config = panel._build_tracking_config()
    assert config.mount_lag_s == 0.0


def test_on_arm_click_arms_immediately_without_a_confirmation_dialog(panel):
    assert panel._armed is False
    panel._on_arm_click()
    assert panel._armed is True
    assert str(panel._start_button["state"]) == "normal"


def test_set_auto_guide_available_enables_and_disables_the_checkbox(panel):
    # The checkbox lives here (Transit tab), not in CalibrationPanel -- it's
    # only useful during an active pass, see CalibrationPanel's
    # on_calibration_ready callback in app.py.
    assert str(panel._auto_guide_check["state"]) == "disabled"
    panel.set_auto_guide_available(True)
    assert str(panel._auto_guide_check["state"]) == "normal"

    panel._auto_guide_var.set(True)
    panel.set_auto_guide_available(False)
    assert str(panel._auto_guide_check["state"]) == "disabled"
    assert panel._auto_guide_var.get() is False


def test_countdown_text_reports_rise_and_set_and_no_window(panel):
    panel._window = None
    assert panel._countdown_text() == ""

    panel._window = _window(datetime.now(timezone.utc) + timedelta(minutes=10))
    assert "Rise in" in panel._countdown_text()

    panel._window = _window(datetime.now(timezone.utc) - timedelta(minutes=10), duration_s=3600.0)
    assert "in progress" in panel._countdown_text()

    panel._window = _window(datetime.now(timezone.utc) - timedelta(hours=2), duration_s=60.0)
    assert "ended" in panel._countdown_text()


def test_perp_nudge_key_flashes_the_matching_button_then_releases_it(panel):
    # Mouse clicks get Tk's own pressed-state animation for free; the
    # keyboard path (arrow keys, see am5/gui/panels.py's TransitPanel
    # binding block) has to trigger it explicitly since no widget was
    # actually clicked -- see _flash_button's docstring.
    assert panel._perp_right_button.instate(["pressed"]) is False
    panel._on_perp_nudge_key(1.0)
    assert panel._perp_right_button.instate(["pressed"]) is True
    assert panel._perp_left_button.instate(["pressed"]) is False

    panel.update()
    time.sleep(GUIDING_PERP_PULSE_DURATION_S + 0.05)
    panel.update()
    assert panel._perp_right_button.instate(["pressed"]) is False


def test_delta_t_key_press_adjusts_offset_and_flashes_the_matching_button(panel):
    dt_before, _ = panel._offsets.snapshot()
    assert panel._delta_t_plus_button.instate(["pressed"]) is False

    panel._on_delta_t_key_press(0.1)
    dt_after, _ = panel._offsets.snapshot()
    assert dt_after == pytest.approx(dt_before + 0.1)
    assert panel._delta_t_plus_button.instate(["pressed"]) is True
    assert panel._delta_t_minus_button.instate(["pressed"]) is False

    panel._on_delta_t_key_press(-0.1)
    dt_final, _ = panel._offsets.snapshot()
    assert dt_final == pytest.approx(dt_before)
    assert panel._delta_t_minus_button.instate(["pressed"]) is True

    panel.update()
    time.sleep(GUIDING_PERP_PULSE_DURATION_S + 0.05)
    panel.update()
    assert panel._delta_t_plus_button.instate(["pressed"]) is False
    assert panel._delta_t_minus_button.instate(["pressed"]) is False


def test_delta_t_arrow_key_works_regardless_of_which_widget_has_focus():
    # Same regression as the perpendicular-nudge case, for the newer ↑ ↓
    # -> delta_t mapping -- see test_perp_nudge_arrow_key_works_regardless_
    # of_which_widget_has_focus's comment.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    try:
        p = TransitPanel(root, mount_worker, camera_worker, Path("/tmp"))
        p.pack()
        root.deiconify()
        root.update()

        captured = []
        p._offsets.adjust_delta_t = lambda step: captured.append(step)

        p._arm_button.focus_force()
        root.update()
        assert root.focus_get() is p._arm_button

        p._arm_button.event_generate("<Up>")
        root.update()
        assert captured == [0.1]
        assert p._delta_t_plus_button.instate(["pressed"]) is True
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_perp_nudge_arrow_key_works_regardless_of_which_widget_has_focus():
    # Regression: a binding placed on the containing Frame does NOT fire
    # just because some descendant widget happens to have focus (Tk only
    # consults the actually-focused widget's own bindtags) -- so as soon
    # as the operator clicked ARM (or any widget besides the preview
    # canvas), arrow-key nudging used to silently stop doing anything.
    # _bind_offset_keys binds recursively on every widget instead.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    try:
        p = TransitPanel(root, mount_worker, camera_worker, Path("/tmp"))
        p.pack()
        root.deiconify()
        root.update()

        fired = []
        p._offsets.trigger_perp_pulse = lambda sign, duration_s=0.15: fired.append(sign)

        p._arm_button.focus_force()
        root.update()
        assert root.focus_get() is p._arm_button

        p._arm_button.event_generate("<Right>")
        root.update()
        assert fired == [1.0]
        assert p._perp_right_button.instate(["pressed"]) is True
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_exposure_panel_compute_populates_result_and_preview_image():
    root = tk.Tk()
    root.withdraw()
    try:
        exposure_panel = ExposurePanel(root)

        ts = load.timescale()
        satellite = EarthSatellite(_TLE_LINE1, _TLE_LINE2, "ISS (fixture)", ts)
        site = wgs84.latlon(46.18, 6.14)
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        window = find_next_pass(satellite, site, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)
        trajectory = compute_trajectory(satellite, site, window.t_rise, window.t_set, step_s=0.2)

        exposure_panel.set_pass(trajectory, window)
        exposure_panel._on_compute_click()  # must not raise -- regression test for a stale self._preview_canvas reference
        root.update()

        assert "Suggested starting gain" in exposure_panel._result_var.get()
        assert "camera px" in exposure_panel._preview_caption_var.get()
        assert exposure_panel._preview_image is not None
    finally:
        root.destroy()


def test_passes_panel_draws_sky_map_without_raising():
    root = tk.Tk()
    root.withdraw()
    try:
        passes_panel = PassesPanel(root, lambda *a: None)

        ts = load.timescale()
        satellite = EarthSatellite(_TLE_LINE1, _TLE_LINE2, "ISS (fixture)", ts)
        site = wgs84.latlon(46.18, 6.14)
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        window = find_next_pass(satellite, site, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)
        trajectory = compute_trajectory(satellite, site, window.t_rise, window.t_set, step_s=0.2)
        crossings = meridian_crossings(trajectory)

        passes_panel._site = site  # required for the constellation background, see _draw_constellations
        passes_panel._draw_sky_map(trajectory, window, crossings)  # must not raise
        root.update()

        # the ISS track itself plus at least one constellation line
        assert len(passes_panel._sky_map.ax.lines) > 1
    finally:
        root.destroy()


def test_passes_panel_defaults_to_iss_with_the_custom_field_disabled():
    root = tk.Tk()
    root.withdraw()
    try:
        p = PassesPanel(root, lambda *a: None)
        assert p._target_var.get() == "ISS (ZARYA)"
        assert str(p._custom_catnr_entry["state"]) == "disabled"
        assert p._resolve_target() == KNOWN_SATELLITES["ISS (ZARYA)"]
    finally:
        root.destroy()


def test_passes_panel_selecting_custom_enables_the_norad_id_field():
    root = tk.Tk()
    root.withdraw()
    try:
        p = PassesPanel(root, lambda *a: None)
        p._target_var.set(CUSTOM_SATELLITE_LABEL)
        root.update()
        assert str(p._custom_catnr_entry["state"]) == "normal"

        # no ID typed yet -- invalid
        assert p._resolve_target() is None

        p._custom_catnr_var.set("48274")
        assert p._resolve_target() == (48274, None)  # non-curated satellites get no magnitude estimate
    finally:
        root.destroy()


def test_passes_panel_table_shows_na_for_an_unestimated_magnitude():
    root = tk.Tk()
    root.withdraw()
    try:
        p = PassesPanel(root, lambda *a: None)
        p._passes = [_window(datetime.now(timezone.utc), magnitude_estimate=float("nan"))]
        p._populate_tree()
        row = p._tree.item("0")["values"]
        assert row[4] == "N/A"
    finally:
        root.destroy()


def test_transit_panel_set_trajectory_draws_sky_map_and_tracks_mount(panel):
    ts = load.timescale()
    satellite = EarthSatellite(_TLE_LINE1, _TLE_LINE2, "ISS (fixture)", ts)
    site = wgs84.latlon(46.18, 6.14)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = find_next_pass(satellite, site, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)
    trajectory = compute_trajectory(satellite, site, window.t_rise, window.t_set, step_s=0.2)
    crossings = meridian_crossings(trajectory)

    panel.set_trajectory(trajectory, window, crossings, site)
    assert panel._site is site
    assert len(panel._sky_map.ax.lines) > 1  # track + constellations drawn

    assert panel._sky_map._mount_marker is None
    panel._update_mount_marker(ra_hours=12.0, dec_deg=45.0)
    marker = panel._sky_map._mount_marker
    assert marker is not None
    first_x, first_y = (float(v) for v in marker.get_data()[0]), (float(v) for v in marker.get_data()[1])
    first_xy = (list(first_x)[0], list(first_y)[0])

    # a different position must move the same artist (set_data), not create a new one
    panel._update_mount_marker(ra_hours=6.0, dec_deg=-10.0)
    assert panel._sky_map._mount_marker is marker
    second_xy = (float(marker.get_data()[0][0]), float(marker.get_data()[1][0]))
    assert second_xy != first_xy


def test_sanitize_filename_collapses_unsafe_characters():
    assert _sanitize_filename("ISS (ZARYA)") == "ISS_ZARYA"
    assert _sanitize_filename("  leading/trailing  ") == "leading_trailing"
    assert _sanitize_filename("a//b") == "a_b"


def _select_a_pass(panel):
    ts = load.timescale()
    satellite = EarthSatellite(_TLE_LINE1, _TLE_LINE2, "ISS (ZARYA)", ts)
    site = wgs84.latlon(46.18, 6.14)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = find_next_pass(satellite, site, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)
    trajectory = compute_trajectory(satellite, site, window.t_rise, window.t_set, step_s=0.2)
    crossings = meridian_crossings(trajectory)
    panel.set_trajectory(trajectory, window, crossings, site, satellite.name)
    return window


def test_recording_with_no_pass_selected_uses_the_flat_out_dir(panel, tmp_path):
    assert panel._window is None
    captured = {}
    panel._camera_worker.start_recording = lambda path, **kw: captured.update(path=path)
    panel._on_toggle_recording()
    assert captured["path"].parent == tmp_path.resolve()


def test_recording_with_a_pass_selected_creates_a_dedicated_folder(panel, tmp_path):
    window = _select_a_pass(panel)
    captured = {}
    panel._camera_worker.start_recording = lambda path, **kw: captured.update(path=path)
    panel._on_toggle_recording()

    expected_name = f"ISS_ZARYA_{window.t_rise.strftime('%Y%m%dT%H%M%S')}"
    pass_dir = tmp_path / expected_name
    assert captured["path"].parent == pass_dir.resolve()
    assert (pass_dir / "pass_info.txt").exists()
    assert "ISS (ZARYA)" in (pass_dir / "pass_info.txt").read_text()
    assert (pass_dir / "skymap.png").exists()
    assert (pass_dir / "skymap.png").stat().st_size > 0
    # per-recording FireCapture-style settings sidecar, matching the .ser name
    sidecar = captured["path"].with_suffix(".txt")
    assert sidecar.exists()
    text = sidecar.read_text()
    assert "Gain:" in text and "Exposure:" in text and "ROI:" in text


def test_two_recordings_of_the_same_pass_share_one_folder(panel):
    _select_a_pass(panel)
    paths = []
    panel._camera_worker.start_recording = lambda path, **kw: paths.append(path)
    panel._on_toggle_recording()
    time.sleep(1.1)  # capture_<timestamp>.ser has 1s resolution -- ensure a distinct filename
    panel._on_toggle_recording()

    assert len(paths) == 2
    assert paths[0].parent == paths[1].parent
    assert paths[0] != paths[1]


def test_snapshot_with_a_pass_selected_lands_in_the_pass_folder(panel, tmp_path):
    window = _select_a_pass(panel)
    captured = {}
    panel._camera_worker.save_fits_snapshot = lambda path, **kw: captured.update(path=path)
    panel._on_snapshot_click()

    expected_name = f"ISS_ZARYA_{window.t_rise.strftime('%Y%m%dT%H%M%S')}"
    assert captured["path"].parent == (tmp_path / expected_name).resolve()


def test_capture_settings_sidecar_reflects_current_gain_and_exposure(panel, tmp_path):
    panel._camera_vars.gain.set(250.0)
    import math
    panel._camera_vars.exposure_log.set(math.log10(5000))
    sidecar = tmp_path / "settings.txt"
    panel._write_capture_settings(sidecar)
    text = sidecar.read_text()
    assert "Gain: 250" in text
    assert "5000" in text


def test_rehearsal_redraw_aligns_rise_marker_with_a_goto_to_the_same_radec(panel):
    # Reproduces the incident: a pass far in the future (real Rise az/alt
    # computed for that future time) must not be compared against a
    # "now"-based telescope marker without also recomputing the Rise
    # marker at "now" -- _on_jog_goto_click's rehearsal redraw is exactly
    # what reconciles the two.
    ts = load.timescale()
    satellite = EarthSatellite(_TLE_LINE1, _TLE_LINE2, "ISS (fixture)", ts)
    site = wgs84.latlon(46.18, 6.14)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = find_next_pass(satellite, site, t0=t0, horizon_deg=10.0, lookahead_hours=48.0)
    trajectory = compute_trajectory(satellite, site, window.t_rise, window.t_set, step_s=0.2)
    crossings = meridian_crossings(trajectory)
    panel.set_trajectory(trajectory, window, crossings, site)

    # Manual GOTO's actual target: the trajectory's very first sample (Rise)
    start_ra_deg, start_dec_deg, _, _ = trajectory.interpolate(float(trajectory.t_unix[0]))

    now = datetime.now(timezone.utc)
    panel._redraw_sky_map(rehearsal_now=now)  # what _on_jog_goto_click does before commanding motion
    rise_line = next(line for line in panel._sky_map.ax.lines if line.get_label() == "Rise")
    rise_marker_r = rise_line.get_data()[1][0]  # r = 90 - alt

    panel._update_mount_marker(ra_hours=(start_ra_deg % 360.0) / 15.0, dec_deg=start_dec_deg)
    telescope_r = panel._sky_map._mount_marker.get_data()[1][0]

    # small tolerance: _redraw_sky_map and _update_mount_marker each call
    # datetime.now() independently, a fraction of a millisecond apart
    assert telescope_r == pytest.approx(rise_marker_r, abs=1e-3)


@pytest.fixture
def calibration_panel():
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    p = CalibrationPanel(root, mount_worker, camera_worker, LiveOffsets())
    p.calibration_ready_calls = []
    p._on_calibration_ready = lambda: p.calibration_ready_calls.append(True)
    yield p
    mount_worker.shutdown()
    camera_worker.shutdown()
    root.destroy()


def _blob(x: float, y: float, found: bool = True) -> BlobDetection:
    return BlobDetection(found=found, centroid_x=x, centroid_y=y, peak_value=200.0, pixel_count=20)


def test_calibration_panel_calibrate_refuses_without_a_detected_blob(calibration_panel):
    calibration_panel._latest_radec = (3.0, 45.0)
    calibration_panel._latest_blob = None
    calibration_panel._on_calibrate_click()
    assert calibration_panel._calib_step is None
    assert "no bright object" in calibration_panel._calib_status_var.get()


def test_calibration_panel_calibration_sequence_produces_valid_matrix(calibration_panel):
    p = calibration_panel
    p._latest_radec = (3.0, 45.0)
    p._latest_blob = _blob(100.0, 100.0)

    p._on_calibrate_click()  # starts the RA nudge (jog_start queued on the worker)
    assert p._calib_step == "ra"
    assert str(p._calibrate_button["state"]) == "disabled"

    # bypass the real self.after(1.5s) timer -- simulate the nudge's result directly
    p._latest_radec = (3.0 + (10.0 / 3600.0) / 15.0, 45.0)  # +10 arcsec RA
    p._latest_blob = _blob(140.0, 100.0)
    p._calib_ra_measure()
    assert p._calib_step == "dec"
    assert p._calib_ra_result is not None

    p._latest_radec = (p._latest_radec[0], 45.0 + 8.0 / 3600.0)  # +8 arcsec DEC
    p._latest_blob = _blob(140.0, 130.0)
    p._calib_dec_measure()

    assert p._calib_step is None
    assert p._calibration is not None
    assert p._calibration.arcsec_per_pixel > 0
    assert p.calibration_ready_calls == [True]
    assert "Calibrated" in p._calib_status_var.get()


def test_calibration_panel_calibrate_uses_the_user_configured_nudge_rate_and_duration(calibration_panel):
    # Narrow FOV (long focal length) can push the target out of frame at
    # the defaults -- the operator needs to be able to dial the nudge down,
    # see GUIDING_CALIB_NUDGE_RATE_X's comment.
    p = calibration_panel
    p._latest_radec = (3.0, 45.0)
    p._latest_blob = _blob(100.0, 100.0)
    p._calib_rate_var.set("2.5")
    p._calib_duration_var.set("0.2")

    captured = {}
    p._mount_worker.jog_start = lambda direction, rate_x: captured.update(direction=direction, rate_x=rate_x)
    p._on_calibrate_click()

    assert captured["direction"] == "e"
    assert captured["rate_x"] == pytest.approx(2.5)


def test_calibration_panel_calibrate_falls_back_to_defaults_on_invalid_rate_or_duration(calibration_panel):
    p = calibration_panel
    p._latest_radec = (3.0, 45.0)
    p._latest_blob = _blob(100.0, 100.0)
    p._calib_rate_var.set("not a number")
    p._calib_duration_var.set("")

    captured = {}
    p._mount_worker.jog_start = lambda direction, rate_x: captured.update(rate_x=rate_x)
    p._on_calibrate_click()

    assert captured["rate_x"] == pytest.approx(GUIDING_CALIB_NUDGE_RATE_X)
    assert p._calib_duration_s() == pytest.approx(GUIDING_CALIB_NUDGE_DURATION_S)


def test_calibration_panel_calibration_aborts_if_blob_lost_mid_sequence(calibration_panel):
    p = calibration_panel
    p.set_connected(True)
    p._latest_radec = (3.0, 45.0)
    p._latest_blob = _blob(100.0, 100.0)
    p._on_calibrate_click()

    p._latest_blob = _blob(0.0, 0.0, found=False)
    p._calib_ra_measure()

    assert p._calib_step is None
    assert p._calibration is None
    assert "lost the blob" in p._calib_status_var.get()
    assert str(p._calibrate_button["state"]) == "normal"


def test_calibration_panel_calibration_fails_cleanly_if_mount_did_not_move(calibration_panel):
    p = calibration_panel
    p._latest_radec = (3.0, 45.0)
    p._latest_blob = _blob(100.0, 100.0)
    p._on_calibrate_click()

    p._latest_radec = (3.0, 45.0)  # no change -- as if the mount never actually moved
    p._latest_blob = _blob(100.0, 100.0)
    p._calib_ra_measure()

    assert p._calib_step is None
    assert p._calibration is None
    assert "didn't move measurably" in p._calib_status_var.get()


def _guiding_trajectory():
    # velocity has both RA and DEC components on purpose -- an offset purely
    # along one calibration axis must not accidentally land purely
    # along-track (zero cross-track) just because the test trajectory's
    # own velocity happens to be aligned with that axis.
    n = 100
    t_unix = time.time() + np.linspace(-50, 50, n)
    return Trajectory(
        t_unix=t_unix, ra_deg=np.full(n, 45.0), dec_deg=np.full(n, 45.0),
        dra_dt_deg_s=np.full(n, 0.01), ddec_dt_deg_s=np.full(n, 0.005),
        alt_deg=np.full(n, 45.0), az_deg=np.full(n, 180.0), ha_hours=np.zeros(n),
        distance_km=np.full(n, 500.0),
    )


def test_calibration_panel_auto_guide_corrects_when_blob_off_center(calibration_panel):
    p = calibration_panel
    p._calibration = GuidingCalibration(
        px_per_ra_arcsec_x=2.0, px_per_ra_arcsec_y=0.0, px_per_dec_arcsec_x=0.0, px_per_dec_arcsec_y=2.0,
    )
    p._trajectory = _guiding_trajectory()

    _, perp_before = p._live_offsets.snapshot()
    assert perp_before == 0.0
    p._maybe_apply_auto_guide_correction((480, 640), _blob(640 / 2 + 50, 480 / 2))
    _, perp_after = p._live_offsets.snapshot()
    assert perp_after != 0.0


def test_calibration_panel_auto_guide_ignores_small_offset_within_deadband(calibration_panel):
    p = calibration_panel
    p._calibration = GuidingCalibration(
        px_per_ra_arcsec_x=2.0, px_per_ra_arcsec_y=0.0, px_per_dec_arcsec_x=0.0, px_per_dec_arcsec_y=2.0,
    )
    p._trajectory = _guiding_trajectory()

    p._maybe_apply_auto_guide_correction((480, 640), _blob(640 / 2 + 1, 480 / 2))
    _, perp = p._live_offsets.snapshot()
    assert perp == 0.0


def test_calibration_panel_handle_camera_event_updates_blob_and_preview(calibration_panel):
    p = calibration_panel
    frame = np.full((60, 80), 15, dtype=np.uint8)
    frame[25:35, 55:65] = 220  # a bright synthetic ISS blob
    p.handle_camera_event(CameraEvent(kind="preview_frame", payload={"pgm": frame_to_pgm(frame), "width": 80, "height": 60}))
    assert p._latest_blob is not None
    assert p._latest_blob.found is True
    assert p._preview_image is not None
    assert "ISS at pixel" in p._blob_status_var.get()


def test_calibration_panel_axis_calibration_done_updates_status(calibration_panel):
    p = calibration_panel
    p.handle_mount_event(WorkerEvent("calibration_done", {"ra_sign": -1.0, "dec_sign": 1.0, "pier_side": "E"}))
    assert "RA sign: -1" in p._axis_calibration_var.get()
    assert "DEC sign: +1" in p._axis_calibration_var.get()
    assert "pier side E" in p._axis_calibration_var.get()


def test_calibration_panel_status_label_refreshes_on_a_pier_flip_position_event(calibration_panel):
    # Regression: a flip detected by App (see AxisSigns.update_pier_side)
    # mutates the shared axis_signs in place -- this panel must reflect
    # that on the next "position" event, not just show a stale value from
    # the original Calibrate click.
    p = calibration_panel
    p.handle_mount_event(WorkerEvent("calibration_done", {"ra_sign": 1.0, "dec_sign": 1.0, "pier_side": "E"}))
    assert "DEC sign: +1" in p._axis_calibration_var.get()

    p._axis_signs.update_pier_side("W")  # simulates what App does on a real flip
    p.handle_mount_event(WorkerEvent("position", {"ra_hours": 5.0, "dec_deg": 45.0}))
    assert "DEC sign: -1" in p._axis_calibration_var.get()
    assert "pier side W" in p._axis_calibration_var.get()


def test_calibration_panel_position_event_before_any_calibration_does_not_touch_status(calibration_panel):
    p = calibration_panel
    assert p._axis_calibration_var.get() == "Not calibrated this session"
    p.handle_mount_event(WorkerEvent("position", {"ra_hours": 5.0, "dec_deg": 45.0}))
    assert p._axis_calibration_var.get() == "Not calibrated this session"


def test_calibration_panel_motion_buttons_disabled_while_parked(calibration_panel):
    # Regression: clicking "Measure mount lag" while parked used to leave
    # the button disabled forever -- _handle_measure_mount_lag is
    # blocked_while_parked on the worker side and never emits the
    # mount_lag_result event the click handler was waiting for to
    # re-enable it. Greying these out while parked prevents that click
    # from ever happening. Same for axis calibrate and camera-to-sky
    # calibrate (also jog-based, also blocked_while_parked).
    p = calibration_panel
    p.set_connected(True)
    assert str(p._lag_measure_button["state"]) == "normal"
    assert str(p._axis_calibrate_button["state"]) == "normal"
    assert str(p._calibrate_button["state"]) == "normal"

    p.handle_mount_event(WorkerEvent("parked", {"method": "home", "reply": None}))
    assert str(p._lag_measure_button["state"]) == "disabled"
    assert str(p._axis_calibrate_button["state"]) == "disabled"
    assert str(p._calibrate_button["state"]) == "disabled"
    # Not mount-related at all -- stays usable regardless.
    assert str(p._clock_sync_button["state"]) == "normal"

    p.handle_mount_event(WorkerEvent("unparked", {}))
    assert str(p._lag_measure_button["state"]) == "normal"
    assert str(p._axis_calibrate_button["state"]) == "normal"
    assert str(p._calibrate_button["state"]) == "normal"


def test_calibration_panel_mount_lag_result_updates_status_and_shared_var(calibration_panel):
    p = calibration_panel
    p.set_connected(True)
    p._lag_measure_button.configure(state="disabled")
    p.handle_mount_event(WorkerEvent("mount_lag_result", {"lag_s": 0.345, "steady_rate_arcsec_s": 1200.0, "samples": 30}))
    assert "0.345" in p._lag_status_var.get()
    assert str(p._lag_measure_button["state"]) == "normal"
    assert p._mount_lag_var.get() == pytest.approx(0.345)


def test_calibration_panel_measure_lag_click_rejects_invalid_input(calibration_panel):
    p = calibration_panel
    p._lag_rate_var.set("not-a-number")
    p._on_measure_lag_click()
    assert "Invalid" in p._lag_status_var.get()


def test_calibration_panel_measure_lag_click_disables_button_and_queues_worker_command(calibration_panel):
    p = calibration_panel
    p._lag_rate_var.set("80")
    p._lag_duration_var.set("1.0")
    p._on_measure_lag_click()
    assert str(p._lag_measure_button["state"]) == "disabled"
    assert "Measuring" in p._lag_status_var.get()


def test_calibration_panel_clock_sync_poll_reports_synchronized_status(calibration_panel):
    p = calibration_panel
    p._clock_sync_button.configure(state="disabled")
    p._clock_sync_results.put(ClockSyncStatus(synchronized=True, offset_s=-0.006451, source="timedatectl timesync-status", detail="..."))
    p._poll_clock_sync_results()
    assert "Synchronized" in p._clock_sync_var.get()
    assert str(p._clock_sync_button["state"]) == "normal"


def test_calibration_panel_clock_sync_poll_reports_unsynchronized_status(calibration_panel):
    p = calibration_panel
    p._clock_sync_results.put(ClockSyncStatus(synchronized=False, offset_s=2.5, source="chronyc tracking", detail="..."))
    p._poll_clock_sync_results()
    assert "NOT synchronized" in p._clock_sync_var.get()


def test_calibration_panel_clock_sync_poll_reports_unknown_status(calibration_panel):
    p = calibration_panel
    p._clock_sync_results.put(ClockSyncStatus(synchronized=None, offset_s=None, source="none", detail="no tool available"))
    p._poll_clock_sync_results()
    assert "Unknown" in p._clock_sync_var.get()
    assert "no tool available" in p._clock_sync_var.get()


def test_camera_connect_uses_the_real_configured_plate_scale():
    # Camera connection lives in ConnectionPanel (moved there so both
    # devices are wired up from one tab), not TransitPanel.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    try:
        train = OpticalTrain(aperture_mm=200, focal_length_mm=1000, barlow_multiplier=2.0, pixel_size_um=2.9)
        p = ConnectionPanel(root, mount_worker, camera_worker, lambda _connected: None, get_optical_train=lambda: train, map_widget_cls=_StubMapWidget)
        captured = {}
        p._camera_worker.connect = lambda *args, **kwargs: captured.update(kwargs, kind=args[0])
        p._on_camera_connect_click()
        assert captured["plate_scale_arcsec_per_px"] == pytest.approx(train.plate_scale_arcsec_per_px)
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_camera_connect_falls_back_to_default_plate_scale_when_fields_invalid():
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    try:
        p = ConnectionPanel(root, mount_worker, camera_worker, lambda _connected: None, get_optical_train=lambda: None, map_widget_cls=_StubMapWidget)
        captured = {}
        p._camera_worker.connect = lambda *args, **kwargs: captured.update(kwargs, kind=args[0])
        p._on_camera_connect_click()
        assert captured["plate_scale_arcsec_per_px"] is None  # CameraWorker/MockAsiCamera fall back to their own default
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_mount_connect_sends_the_configured_site_lat_lon():
    # Regression: connect() used to always default to a hardcoded Geneva
    # coordinate regardless of what the operator configured -- see
    # SiteVars' docstring.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    try:
        site_vars = SiteVars.create()
        site_vars.lat.set("48.8589")
        site_vars.lon.set("2.3200")
        p = ConnectionPanel(root, mount_worker, camera_worker, lambda _connected: None, site_vars=site_vars, map_widget_cls=_StubMapWidget)
        captured = {}
        p._worker.connect = lambda *args, **kwargs: captured.update(kwargs, kind=args[0])
        p._on_connect_click()
        assert captured["latitude_deg"] == pytest.approx(48.8589)
        assert captured["longitude_deg"] == pytest.approx(2.3200)
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_mount_connect_refuses_invalid_site_lat_lon_without_connecting():
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    try:
        site_vars = SiteVars.create()
        site_vars.lat.set("not a number")
        p = ConnectionPanel(root, mount_worker, camera_worker, lambda _connected: None, site_vars=site_vars, map_widget_cls=_StubMapWidget)
        called = []
        p._worker.connect = lambda *args, **kwargs: called.append((args, kwargs))
        p._on_connect_click()
        assert called == []
        assert "Invalid" in p._status_var.get()
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_apply_location_updates_shared_site_vars_and_status():
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    try:
        site_vars = SiteVars.create()
        p = ConnectionPanel(root, mount_worker, camera_worker, lambda _connected: None, site_vars=site_vars, map_widget_cls=_StubMapWidget)
        p._apply_location(48.8566, 2.3522, label="Paris, France")
        assert site_vars.lat.get() == "48.8566"
        assert site_vars.lon.get() == "2.3522"
        assert p._map_status_var.get() == "Paris, France"
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_connection_and_passes_panels_share_the_same_site_vars():
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    try:
        site_vars = SiteVars.create()
        connection_panel = ConnectionPanel(root, mount_worker, camera_worker, lambda _connected: None, site_vars=site_vars, map_widget_cls=_StubMapWidget)
        passes_panel = PassesPanel(root, lambda *a: None, site_vars=site_vars)
        connection_panel._apply_location(48.8566, 2.3522)
        assert passes_panel._site_vars.lat.get() == "48.8566"
        assert passes_panel._site_vars.lon.get() == "2.3522"
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_passes_panel_uses_the_configured_site_elevation():
    # elevation_m is Passes-tab-only (see SiteVars' docstring -- the mount
    # protocol has no altitude command), fed to Skyfield's wgs84.latlon for
    # more accurate rise/set timing at non-zero altitude.
    root = tk.Tk()
    root.withdraw()
    try:
        site_vars = SiteVars.create()
        site_vars.lat.set("46.5")
        site_vars.lon.set("7.0")
        site_vars.elevation_m.set("1800")
        p = PassesPanel(root, lambda *a: None, site_vars=site_vars)
        captured = {}
        p._fetch_and_find = lambda lat, lon, elevation_m, *rest: captured.update(
            lat=lat, lon=lon, elevation_m=elevation_m,
        )
        p._on_refresh_click()
        assert captured["lat"] == pytest.approx(46.5)
        assert captured["lon"] == pytest.approx(7.0)
        assert captured["elevation_m"] == pytest.approx(1800.0)
    finally:
        root.destroy()


def test_passes_panel_refuses_invalid_elevation_without_fetching():
    root = tk.Tk()
    root.withdraw()
    try:
        site_vars = SiteVars.create()
        site_vars.elevation_m.set("not a number")
        p = PassesPanel(root, lambda *a: None, site_vars=site_vars)
        called = []
        p._fetch_and_find = lambda *args, **kwargs: called.append((args, kwargs))
        p._on_refresh_click()
        assert called == []
        assert "Invalid" in p._detail_var.get()
    finally:
        root.destroy()


def test_passes_panel_fetch_and_find_passes_elevation_to_wgs84_latlon():
    # More direct check that elevation actually reaches Skyfield's site
    # object, not just that _on_refresh_click forwards the right number.
    root = tk.Tk()
    root.withdraw()
    try:
        site_vars = SiteVars.create()
        p = PassesPanel(root, lambda *a: None, site_vars=site_vars)
        captured = {}
        with patch("am5.gui.panels.load_satellite_tle", return_value=object()):
            with patch("am5.gui.panels.wgs84.latlon", side_effect=lambda *a, **kw: captured.update(kwargs=kw)):
                with patch("am5.gui.panels.find_passes", side_effect=Exception("stop before propagating")):
                    p._fetch_and_find(46.5, 7.0, 1800.0, 10.0, 48.0, 25544, -1.8)
        assert captured["kwargs"]["elevation_m"] == pytest.approx(1800.0)
    finally:
        root.destroy()
