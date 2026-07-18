import math
import threading
import time
import tkinter as tk
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from skyfield.api import EarthSatellite, load, wgs84

import am5.gui.panels as gui_panels
from am5.clock_sync import ClockSyncStatus
from am5.ephemeris import PassWindow, Trajectory, compute_trajectory, find_next_pass, meridian_crossings
from am5.gui.panels import (
    AUTO_GOTO_BEFORE_START_THRESHOLD_DEG,
    CUSTOM_SATELLITE_LABEL,
    GUIDING_CALIB_NUDGE_DURATION_S,
    GUIDING_CALIB_NUDGE_RATE_X,
    GUIDING_PERP_PULSE_DURATION_S,
    KNOWN_SATELLITES,
    MAX_CALIBRATION_PREVIEW_DIM,
    MAX_TRACKING_DURATION_S,
    POLAR_SOLVE_RETRY_ATTEMPTS,
    AlignmentPanel,
    AlignmentSkyMapWidget,
    CameraControlVars,
    ConnectionPanel,
    ExposurePanel,
    CalibrationPanel,
    FinderCameraPanel,
    PALETTE,
    PassesPanel,
    SiteVars,
    TransitPanel,
    _local_and_utc,
    _meridian_detail_line,
    _normalize_to_8bit_for_preview,
    _sanitize_filename,
    _scale_16bit_to_8bit_fixed,
    visible_named_stars,
)
from am5.gui.finder_window import FinderWindow
from am5.gui.worker import MountWorker, WorkerEvent
from am5.optics import OpticalTrain
from am5.tracker import AxisSigns, LiveOffsets, _perp_rate_components
from camera.finder import FinderCalibration, FinderState
from camera.guiding import BlobDetection, GuidingCalibration, calibrate_from_nudges
from camera.worker import CameraEvent, CameraWorker, frame_to_pgm

# Same fixed, network-free TLE as tests/test_ephemeris.py.
_TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9006"
_TLE_LINE2 = "2 25544  51.6400 208.9163 0006317  69.9862 25.2825 15.49560500000000"


def test_scale_16bit_to_8bit_fixed_a_uniform_gain_change_stays_visible():
    # Regression: show_frame_on_canvas (FinderWindow's live preview, and
    # AlignmentPanel's polar-alignment preview) used to share
    # _normalize_to_8bit_for_preview with the SER player -- that function
    # rescales EVERY frame so its own max always maps to 255, which
    # exactly cancels out a uniform gain change algebraically (k*frame /
    # (k*max) == frame/max). Reported symptom: gain slider had no visible
    # effect on the live preview. A FIXED reference point (this project's
    # own camera's real 12-bit ADC range) must not have this
    # cancellation -- scaling the raw frame by a real gain-like factor
    # has to change the OUTPUT.
    base = np.full((10, 10), 200, dtype=np.uint16)  # well under the 4095 12-bit ceiling, no clipping
    low_gain = _scale_16bit_to_8bit_fixed(base)
    high_gain = _scale_16bit_to_8bit_fixed(base * 2)  # simulates ~2x more signal from more gain
    assert int(high_gain[0, 0]) > int(low_gain[0, 0])


def test_scale_16bit_to_8bit_fixed_passes_uint8_values_through_unchanged_at_1x():
    frame = np.array([[10, 250]], dtype=np.uint8)
    result = _scale_16bit_to_8bit_fixed(frame)
    np.testing.assert_array_equal(result, frame)


def test_scale_16bit_to_8bit_fixed_manual_stretch_boosts_uint8_too():
    # The manual "Preview stretch" slider (FinderWindow) is purely
    # display-side and applies regardless of the camera's own bit depth,
    # unlike the fixed 12-bit ADC conversion above (which only matters
    # for RAW16 frames).
    frame = np.array([[10, 100]], dtype=np.uint8)
    result = _scale_16bit_to_8bit_fixed(frame, stretch=2.0)
    np.testing.assert_array_equal(result, np.array([[20, 200]], dtype=np.uint8))


def test_normalize_to_8bit_for_preview_still_auto_stretches_for_ser_playback():
    # SerPlayerPanel's own use of this function is unaffected by the fix
    # above -- SER's PixelDepth can legitimately be anything up to 16, so
    # per-frame auto-stretch is still the right (if imperfect) behavior
    # there, see the function's own docstring.
    frame = np.full((4, 4), 100, dtype=np.uint16)
    result = _normalize_to_8bit_for_preview(frame)
    assert int(result.max()) == 255


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


def test_tracking_tick_event_does_not_crash_the_event_pump(panel):
    # Regression: _maybe_apply_finder_correction used to be defined on
    # CalibrationPanel (which has neither self._finder_state nor
    # self._finder_correct_var -- both TransitPanel-only), while its only
    # call site was already correctly here, in TransitPanel.
    # handle_mount_event's "tracking_tick" branch -- an AttributeError on
    # every single real tracking_tick (about 1s after any tracking session
    # starts). In the real app this propagates out of App._pump_events
    # BEFORE it reaches its own self.root.after(...) reschedule call at
    # the very end, permanently killing the whole event pump -- not just
    # this panel -- for the rest of the session (no more camera previews,
    # no more mount position updates, nothing). This is what a reported
    # "the app just freezes right after I click Simulate" was actually
    # tracked back to.
    event = WorkerEvent("tracking_tick", {
        "actual_ra_deg": 10.0, "actual_dec_deg": 20.0, "target_ra_deg": 10.0, "target_dec_deg": 20.0,
        "elapsed_s": 1.0, "along_track_arcsec": 0.0, "cross_track_arcsec": 0.0,
    })
    panel.handle_mount_event(event)  # must not raise


def test_maybe_apply_finder_correction_requires_the_main_cameras_own_sky_calibration(panel):
    # A finder-to-main geometric calibration alone isn't enough -- see
    # FinderState.main_calibration's own field docstring. Without it,
    # get_correction_arcsec returns None and nothing should trigger, even
    # with a blob locked and the checkbox on.
    state = FinderState()
    state.calibration = FinderCalibration(calibrated=True)
    state.blob_found = True
    state.last_blob_row = 100.0
    state.last_blob_col = 250.0
    state.last_frame = np.zeros((200, 300), dtype=np.uint8)
    panel._finder_state = state
    panel._finder_correct_var.set(True)
    panel._active_trajectory = _guiding_trajectory()

    triggered = []
    panel._offsets.trigger_perp_pulse = lambda sign, **kw: triggered.append(sign)

    panel._maybe_apply_finder_correction()

    assert triggered == []


def test_maybe_apply_finder_correction_points_toward_the_target_not_away(panel):
    # Same regression class as CalibrationPanel._maybe_apply_auto_guide_
    # correction's own fix: verify the DIRECTION of the correction, not
    # just that something fired. Chains a real finder-to-main calibration
    # (identity: offset=0, ratio=1, rotation=0, so finder px == main px)
    # through a real nudge-derived main-camera GuidingCalibration (same
    # setup as the auto-guide regression test), and confirms the resulting
    # perp pulse's rate correction actually reduces -- not increases -- the
    # boresight's lag behind the target.
    state = FinderState()
    state.calibration = FinderCalibration(calibrated=True)
    state.blob_found = True
    finder_h, finder_w = 200, 300
    state.last_frame = np.zeros((finder_h, finder_w), dtype=np.uint8)
    # 10px east of centre -- with an identity finder-to-main mapping, this
    # is exactly a main-camera dx_px=10, dy_px=0 offset from main centre.
    state.last_blob_row = finder_h / 2.0
    state.last_blob_col = finder_w / 2.0 + 10.0
    state.set_main_calibration(calibrate_from_nudges(10.0, -10.0, 0.0, 10.0, 0.0, -10.0))
    panel._finder_state = state
    panel._finder_correct_var.set(True)

    n = 10
    t_unix = time.time() + np.linspace(-5, 5, n)
    panel._active_trajectory = Trajectory(
        t_unix=t_unix, ra_deg=np.zeros(n), dec_deg=np.zeros(n),
        dra_dt_deg_s=np.zeros(n), ddec_dt_deg_s=np.full(n, 0.001),
        alt_deg=np.full(n, 45.0), az_deg=np.full(n, 180.0), ha_hours=np.zeros(n),
        distance_km=np.full(n, 500.0),
    )

    triggered = []
    panel._offsets.trigger_perp_pulse = lambda sign, **kw: triggered.append(sign)

    panel._maybe_apply_finder_correction()

    assert len(triggered) == 1
    extra_dra_dt, _ = _perp_rate_components(0.0, 0.0, 0.001, triggered[0])
    assert extra_dra_dt > 0, (
        "finder correction points away from the target instead of toward it "
        f"(extra_dra_dt={extra_dra_dt}, should be positive to catch up a lagging boresight)"
    )


def test_maybe_apply_finder_correction_backs_off_once_main_camera_has_a_lock(panel):
    # Regression: finder correction and main-camera auto-guide used to run
    # fully independently -- once the ISS drifted into the main camera's
    # own narrow FOV too, both correctors would fire on the SAME
    # trigger_perp_pulse from different blob detections at the same time,
    # fighting each other instead of a clean acquire-then-track handoff.
    # FinderState.main_blob_locked (set by CalibrationPanel.handle_camera_
    # event) is how the main camera announces "I've got it now" -- finder
    # correction must stand down as soon as it's True, using the exact
    # same otherwise-correcting setup as the sibling "points toward the
    # target" test above.
    state = FinderState()
    state.calibration = FinderCalibration(calibrated=True)
    state.blob_found = True
    finder_h, finder_w = 200, 300
    state.last_frame = np.zeros((finder_h, finder_w), dtype=np.uint8)
    state.last_blob_row = finder_h / 2.0
    state.last_blob_col = finder_w / 2.0 + 10.0
    state.set_main_calibration(calibrate_from_nudges(10.0, -10.0, 0.0, 10.0, 0.0, -10.0))
    state.set_main_blob_locked(True)
    panel._finder_state = state
    panel._finder_correct_var.set(True)

    n = 10
    t_unix = time.time() + np.linspace(-5, 5, n)
    panel._active_trajectory = Trajectory(
        t_unix=t_unix, ra_deg=np.zeros(n), dec_deg=np.zeros(n),
        dra_dt_deg_s=np.zeros(n), ddec_dt_deg_s=np.full(n, 0.001),
        alt_deg=np.full(n, 45.0), az_deg=np.full(n, 180.0), ha_hours=np.zeros(n),
        distance_km=np.full(n, 500.0),
    )

    triggered = []
    panel._offsets.trigger_perp_pulse = lambda sign, **kw: triggered.append(sign)

    panel._maybe_apply_finder_correction()

    assert triggered == []


def test_maybe_apply_finder_correction_skips_outside_the_trajectorys_active_window(panel):
    # Same class of bug as CalibrationPanel's own sibling test -- outside
    # the trajectory's active window, interpolate() zeroes dra_dt/ddec_dt,
    # and decompose_error's zero-speed branch returns an always-non-
    # negative magnitude instead of a signed cross-track error, which used
    # to become an always-negative "correction" regardless of the true
    # error direction. A pass selected well in advance (finder correction
    # doesn't require tracking to have started at all) is exactly this
    # scenario.
    state = FinderState()
    state.calibration = FinderCalibration(calibrated=True)
    state.blob_found = True
    finder_h, finder_w = 200, 300
    state.last_frame = np.zeros((finder_h, finder_w), dtype=np.uint8)
    state.last_blob_row = finder_h / 2.0
    state.last_blob_col = finder_w / 2.0 + 10.0
    state.set_main_calibration(calibrate_from_nudges(10.0, -10.0, 0.0, 10.0, 0.0, -10.0))
    panel._finder_state = state
    panel._finder_correct_var.set(True)

    n = 10
    t_unix = time.time() + np.linspace(100.0, 200.0, n)  # starts 100s in the future
    panel._active_trajectory = Trajectory(
        t_unix=t_unix, ra_deg=np.zeros(n), dec_deg=np.zeros(n),
        dra_dt_deg_s=np.zeros(n), ddec_dt_deg_s=np.full(n, 0.001),
        alt_deg=np.full(n, 45.0), az_deg=np.full(n, 180.0), ha_hours=np.zeros(n),
        distance_km=np.full(n, 500.0),
    )

    triggered = []
    panel._offsets.trigger_perp_pulse = lambda sign, **kw: triggered.append(sign)

    panel._maybe_apply_finder_correction()

    assert triggered == []


def test_calibration_panel_reports_main_blob_lock_only_when_auto_guide_enabled_and_found():
    # The other half of the handoff: CalibrationPanel is the one deciding
    # whether the main camera "has it" -- must only report locked when
    # auto-guiding is BOTH enabled AND actually seeing the ISS, not just
    # blob.found on its own, so finder correction stays in sole control
    # when the operator hasn't turned auto-guiding on at all (even if the
    # main camera happens to also see the ISS).
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    finder_state = FinderState()
    auto_guide_var = tk.BooleanVar(value=False)
    try:
        p = CalibrationPanel(
            root, mount_worker, camera_worker, LiveOffsets(),
            finder_state=finder_state, auto_guide_var=auto_guide_var,
        )
        frame = np.full((60, 80), 15, dtype=np.uint8)
        frame[25:35, 55:65] = 220  # a bright synthetic ISS blob

        p.handle_camera_event(CameraEvent(kind="preview_frame", payload={"pgm": frame_to_pgm(frame), "width": 80, "height": 60}))
        assert finder_state.main_blob_locked is False  # auto-guide still off

        auto_guide_var.set(True)
        p.handle_camera_event(CameraEvent(kind="preview_frame", payload={"pgm": frame_to_pgm(frame), "width": 80, "height": 60}))
        assert finder_state.main_blob_locked is True

        blank = np.full((60, 80), 15, dtype=np.uint8)  # no blob this time
        p.handle_camera_event(CameraEvent(kind="preview_frame", payload={"pgm": frame_to_pgm(blank), "width": 80, "height": 60}))
        assert finder_state.main_blob_locked is False
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_maybe_apply_finder_correction_does_nothing_when_checkbox_unchecked(panel):
    state = FinderState()
    state.calibration = FinderCalibration(calibrated=True)
    state.blob_found = True
    state.last_blob_row = 100.0
    state.last_blob_col = 250.0
    state.last_frame = np.zeros((200, 300), dtype=np.uint8)
    panel._finder_state = state
    panel._finder_correct_var.set(False)

    triggered = []
    panel._offsets.trigger_perp_pulse = lambda sign, **kw: triggered.append(sign)

    panel._maybe_apply_finder_correction()

    assert triggered == []


def test_on_simulate_click_propagates_the_shifted_trajectory_to_calibration_panel(tmp_path):
    # Regression: _on_simulate_click computes its own time-shifted
    # trajectory (real geometry, relabeled to start "now") but used to
    # only ever hand it to the tracking loop itself -- both this panel's
    # own _maybe_apply_finder_correction and CalibrationPanel's auto-guide
    # correction kept reading the ORIGINAL, unshifted, real-future
    # trajectory, which never overlapped real "now" during a Simulate
    # run. Confirmed directly (real mount + mock camera + Simulate track
    # + both correction checkboxes on): no correction ever applied.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    captured = []
    try:
        p = TransitPanel(
            root, mount_worker, camera_worker, tmp_path,
            on_tracking_trajectory_changed=captured.append,
        )
        _select_a_pass(p)  # a real, past-relative-to-now pass -- needs shifting to overlap "now"
        p._mount_worker.start_tracking = lambda *a, **kw: None

        p._on_simulate_click()

        assert len(captured) == 1
        shifted = captured[0]
        assert shifted is not None
        now = time.time()
        assert shifted.t_unix[0] <= now <= shifted.t_unix[-1], (
            "the trajectory propagated to CalibrationPanel doesn't overlap real 'now' -- "
            "it's the original unshifted trajectory, not Simulate's own shifted copy"
        )
    finally:
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


def _trajectory_at(ra_deg: float, dec_deg: float, duration_s: float = 60.0) -> Trajectory:
    n = 10
    t_unix = time.time() + np.linspace(0.0, duration_s, n)
    return Trajectory(
        t_unix=t_unix, ra_deg=np.full(n, ra_deg), dec_deg=np.full(n, dec_deg),
        dra_dt_deg_s=np.zeros(n), ddec_dt_deg_s=np.zeros(n),
        alt_deg=np.full(n, 45.0), az_deg=np.full(n, 180.0), ha_hours=np.zeros(n),
        distance_km=np.full(n, 500.0),
    )


def test_on_start_click_auto_gotos_first_when_far_off_target(panel):
    # Regression, reported live: picking a "Live now" satellite (a real,
    # arbitrary current sky position, unlike a scheduled pass an operator
    # has usually already pointed toward) and going straight ARM -> Start
    # left the mount wherever it was previously -- the tracker's own
    # runaway-divergence check then correctly rejected it ("pointing
    # error 79.8 deg exceeds runaway limit 10.0 deg"). Now Start itself
    # auto-GOTOs to the target first when the last known position is off
    # by more than AUTO_GOTO_BEFORE_START_THRESHOLD_DEG, instead of
    # requiring the operator to click a GOTO button first.
    panel._window = _window(datetime.now(timezone.utc), duration_s=300.0)
    panel._trajectory = _trajectory_at(ra_deg=200.0, dec_deg=60.0)
    panel._armed = True
    panel._last_known_mount_radec = (0.0, 0.0)  # RA=0h DEC=0 -- far from RA=200deg/DEC=60

    goto_calls = []
    panel._mount_worker.goto = lambda ra_hours, dec_deg: goto_calls.append((ra_hours, dec_deg))
    start_calls = []
    panel._mount_worker.start_tracking = lambda *a, **kw: start_calls.append(a)

    panel._on_start_click()

    assert len(goto_calls) == 1
    ra_hours, dec_deg = goto_calls[0]
    assert ra_hours == pytest.approx(200.0 / 15.0, abs=0.01)
    assert dec_deg == pytest.approx(60.0, abs=0.01)
    assert start_calls == []  # tracking must NOT start yet -- waiting on goto_arrived
    assert panel._start_pending_after_goto is True
    assert str(panel._start_button["state"]) == "disabled"
    assert str(panel._arm_button["state"]) == "disabled"

    # goto_arrived resumes into the deferred start.
    panel.handle_mount_event(WorkerEvent("goto_arrived", {}))
    assert len(start_calls) == 1
    assert panel._start_pending_after_goto is False


def test_on_start_click_skips_auto_goto_when_already_close(panel):
    panel._window = _window(datetime.now(timezone.utc), duration_s=300.0)
    panel._trajectory = _trajectory_at(ra_deg=200.0, dec_deg=60.0)
    panel._armed = True
    # Within AUTO_GOTO_BEFORE_START_THRESHOLD_DEG of the target already.
    panel._last_known_mount_radec = (200.0 / 15.0, 60.0 + AUTO_GOTO_BEFORE_START_THRESHOLD_DEG / 2.0)

    goto_calls = []
    panel._mount_worker.goto = lambda *a: goto_calls.append(a)
    start_calls = []
    panel._mount_worker.start_tracking = lambda *a, **kw: start_calls.append(a)

    panel._on_start_click()

    assert goto_calls == []
    assert len(start_calls) == 1


def test_on_start_click_skips_auto_goto_comparison_with_no_known_position(panel):
    # No "position" event has arrived yet -- can't judge divergence, so
    # this falls back to the original (pre-auto-GOTO) behavior: start
    # tracking directly, same as before this feature existed.
    panel._window = _window(datetime.now(timezone.utc), duration_s=60.0)
    panel._trajectory = object()  # never touched -- _goto_start_radec must not be called in this path
    panel._armed = True
    assert panel._last_known_mount_radec is None

    start_calls = []
    panel._mount_worker.start_tracking = lambda *a, **kw: start_calls.append(a)
    panel._on_start_click()
    assert len(start_calls) == 1


def test_auto_goto_timeout_reports_failure_and_does_not_start_tracking(panel):
    panel._window = _window(datetime.now(timezone.utc), duration_s=300.0)
    panel._trajectory = _trajectory_at(ra_deg=200.0, dec_deg=60.0)
    panel._armed = True
    panel._last_known_mount_radec = (0.0, 0.0)
    panel._mount_worker.goto = lambda *a: None
    start_calls = []
    panel._mount_worker.start_tracking = lambda *a, **kw: start_calls.append(a)
    panel._mount_connected = True

    panel._on_start_click()
    assert panel._start_pending_after_goto is True

    panel.handle_mount_event(WorkerEvent("goto_timeout", {}))

    assert start_calls == []
    assert panel._start_pending_after_goto is False
    assert "did not arrive" in panel._tracking_status_var.get()
    assert str(panel._start_button["state"]) == "normal"


def test_disconnect_clears_a_pending_auto_goto_start(panel):
    panel._window = _window(datetime.now(timezone.utc), duration_s=300.0)
    panel._trajectory = _trajectory_at(ra_deg=200.0, dec_deg=60.0)
    panel._armed = True
    panel._last_known_mount_radec = (0.0, 0.0)
    panel._mount_worker.goto = lambda *a: None
    panel._on_start_click()
    assert panel._start_pending_after_goto is True

    panel.set_mount_connected(False)

    assert panel._start_pending_after_goto is False
    assert panel._last_known_mount_radec is None


def test_build_tracking_config_falls_back_to_zero_lag_on_invalid_input(panel):
    panel._mount_lag_var = tk.StringVar(value="not-a-number")  # simulate a bad manual edit
    config = panel._build_tracking_config()
    assert config.mount_lag_s == 0.0


def test_on_arm_click_arms_immediately_without_a_confirmation_dialog(panel):
    assert panel._armed is False
    panel._on_arm_click()
    assert panel._armed is True
    assert str(panel._start_button["state"]) == "normal"


def test_tracking_stopped_disarms_so_start_button_and_armed_flag_agree(panel):
    # Regression: after tracking stopped, Start stayed disabled (its own
    # _on_start_click disabled it and this branch never re-enables it)
    # while _armed stayed True internally -- the flag and the button
    # disagreed, and a later jog_goto_result/goto_result would then
    # silently re-enable Start off that stale True without the operator
    # re-confirming they're on target. Now a stop disarms, so a fresh ARM
    # is required before tracking again.
    _select_a_pass(panel)
    panel.set_mount_connected(True)

    for stop_kind in ("tracking_stopped", "tracking_error"):
        # Reproduce the post-Start-click state directly (arming then
        # clicking Start disables Start) -- _select_a_pass uses a fixed
        # past window, so calling the real _on_start_click here would hit
        # _check_pass_timing's "pass already over" messagebox and block a
        # headless test; the branch under test doesn't depend on how Start
        # got disabled, only on it being disabled with _armed True.
        panel._on_arm_click()
        panel._start_button.configure(state="disabled")
        assert panel._armed is True

        panel.handle_mount_event(WorkerEvent(stop_kind, {}))
        assert panel._armed is False
        assert str(panel._start_button["state"]) == "disabled"

        # A later GOTO completing must NOT re-enable Start off a stale
        # _armed -- it stays disabled until the operator ARMs again.
        panel.handle_mount_event(WorkerEvent("jog_goto_result", {"arrived": True}))
        assert str(panel._start_button["state"]) == "disabled"


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


def test_set_trajectory_resets_a_stale_delta_t_from_a_previous_pass(panel):
    # Regression: delta_t_s used to persist for the whole app session --
    # a clock-offset/along-track correction dialed in by hand during one
    # pass silently carried over and applied from tick one of whatever
    # pass got selected next. Measured impact (see LiveOffsets.reset's
    # own docstring): the ISS moves at ~900-1400 arcsec/s, so even a
    # small leftover delta_t is a real, large along-track offset at the
    # new pass's start -- with nothing warning the operator beyond an
    # easy-to-miss "+X.Xs" label.
    window = _select_a_pass(panel)
    panel._offsets.adjust_delta_t(1.5)
    panel._offsets.trigger_perp_pulse(1.0, duration_s=30.0)  # still "active" if not cleared
    assert panel._offsets.snapshot() == (1.5, 1.0)

    # Selecting a pass again (same or different -- what matters is that
    # set_trajectory is what the app calls on every fresh pass selection)
    # must clear the stale offset.
    ts = load.timescale()
    satellite = EarthSatellite(_TLE_LINE1, _TLE_LINE2, "ISS (fixture)", ts)
    crossings = meridian_crossings(compute_trajectory(satellite, panel._site, window.t_rise, window.t_set, step_s=0.2))
    trajectory = compute_trajectory(satellite, panel._site, window.t_rise, window.t_set, step_s=0.2)
    panel.set_trajectory(trajectory, window, crossings, panel._site, satellite.name)

    assert panel._offsets.snapshot() == (0.0, 0.0)


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


def test_recording_started_event_disables_roi_and_bit_depth_widgets(panel):
    # The worker refuses ROI/bit-depth changes mid-recording (see
    # CameraWorker._handle_set_roi/_handle_set_bit_depth, added after a
    # live change was found to corrupt the SER file being written) -- the
    # GUI must grey these out too, not just rely on the log warning, so
    # the operator isn't clicking a control that silently no-ops.
    for widget in panel._roi_bitdepth_widgets:
        widget.configure(state="normal")

    panel.handle_camera_event(CameraEvent(kind="recording_started", payload={"path": "/tmp/x.ser"}))

    for widget in panel._roi_bitdepth_widgets:
        assert str(widget["state"]) == "disabled"


def test_recording_stopped_event_reenables_roi_and_bit_depth_widgets(panel):
    panel.handle_camera_event(CameraEvent(kind="recording_started", payload={"path": "/tmp/x.ser"}))
    panel.handle_camera_event(CameraEvent(
        kind="recording_stopped",
        payload={"path": "/tmp/x.ser", "frame_count": 10, "buffer_dropped_frames": 0},
    ))

    for widget in panel._roi_bitdepth_widgets:
        expected = "readonly" if widget is panel._bit_depth_combo else "normal"
        assert str(widget["state"]) == expected


def test_recording_stopped_with_an_error_reports_it_and_keeps_the_partial_frame_count(panel):
    # A write-thread failure (disk full, permissions -- see
    # CameraWorker._write_loop) now surfaces an "error" field instead of
    # silently reporting a clean stop; the panel must show it, not just
    # the frame count, so the operator knows the recording was cut short.
    panel.handle_camera_event(CameraEvent(kind="recording_started", payload={"path": "/tmp/x.ser"}))
    panel.handle_camera_event(CameraEvent(
        kind="recording_stopped",
        payload={"path": "/tmp/x.ser", "frame_count": 2, "buffer_dropped_frames": 0, "error": "disk full"},
    ))

    assert "disk full" in panel._path_var.get()
    assert "2" in panel._path_var.get()
    assert str(panel._path_label["foreground"]) == PALETTE.accent_warn


def test_recording_stopped_without_an_error_reports_a_clean_save(panel):
    panel.handle_camera_event(CameraEvent(kind="recording_started", payload={"path": "/tmp/x.ser"}))
    panel.handle_camera_event(CameraEvent(
        kind="recording_stopped",
        payload={"path": "/tmp/x.ser", "frame_count": 42, "buffer_dropped_frames": 0},
    ))

    assert "Saved" in panel._path_var.get()
    assert str(panel._path_label["foreground"]) == PALETTE.accent_ok


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


def test_pass_info_and_skymap_are_only_written_once_per_pass(panel):
    # Regression test: _write_skymap does a synchronous matplotlib
    # savefig() on the Tk main thread (no worker involved) -- redoing it
    # on every single recording/snapshot click during the same pass was a
    # real, avoidable UI stutter risk right when starting a recording
    # during a live, time-critical pass.
    _select_a_pass(panel)
    panel._camera_worker.start_recording = lambda path, **kw: None
    panel._camera_worker.save_fits_snapshot = lambda path, **kw: None

    with patch.object(panel, "_write_pass_info") as write_info, patch.object(panel, "_write_skymap") as write_map:
        panel._on_toggle_recording()  # first call -- must write both
        panel._on_snapshot_click()  # same pass, second call -- must NOT rewrite
        panel._on_toggle_recording()  # third call -- still must not rewrite
        assert write_info.call_count == 1
        assert write_map.call_count == 1


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


def test_jog_goto_click_disables_start_and_simulate_until_it_completes(panel):
    # Regression test for a real incident: Start/Simulate weren't gated on
    # jog_goto's own in-progress state, so clicking one while a jog_goto
    # was still converging queued a start_tracking command that only
    # began once jog_goto finally finished (MountWorker runs one command
    # at a time) -- but Simulate's "start now" time-shift is computed at
    # CLICK time, so tracking began with a large, silent along-track
    # error baked in (~1.36 deg measured on real hardware -- well under
    # the runaway guard's 10 deg threshold, but easily enough to put a
    # narrow-FOV target outside the frame for the whole pass).
    _select_a_pass(panel)
    panel.set_mount_connected(True)
    panel._on_arm_click()
    assert str(panel._start_button["state"]) == "normal"
    assert str(panel._simulate_button["state"]) == "normal"

    panel._mount_worker.jog_goto = lambda *a, **kw: None
    panel._on_jog_goto_click()
    assert str(panel._jog_goto_button["state"]) == "disabled"
    assert str(panel._arm_button["state"]) == "disabled"
    assert str(panel._start_button["state"]) == "disabled"
    assert str(panel._simulate_button["state"]) == "disabled"

    panel.handle_mount_event(WorkerEvent("jog_goto_result", {"arrived": True}))
    assert str(panel._jog_goto_button["state"]) == "normal"
    assert str(panel._arm_button["state"]) == "normal"
    assert str(panel._simulate_button["state"]) == "normal"
    # Was armed before the jog_goto -- must come back enabled, not left
    # disabled just because jog_goto touched it.
    assert str(panel._start_button["state"]) == "normal"


def test_jog_goto_click_does_not_re_enable_start_if_never_armed(panel):
    _select_a_pass(panel)
    panel.set_mount_connected(True)
    assert str(panel._start_button["state"]) == "disabled"  # never armed

    panel._mount_worker.jog_goto = lambda *a, **kw: None
    panel._on_jog_goto_click()
    panel.handle_mount_event(WorkerEvent("jog_goto_result", {"arrived": True}))
    assert str(panel._start_button["state"]) == "disabled"


def test_mount_goto_click_disables_start_and_simulate_until_it_arrives(panel):
    # Sibling bug to test_jog_goto_click_disables_start_and_simulate_
    # until_it_completes above, found via cross-referencing every
    # WorkerEvent kind against every listener: the native mount-GOTO
    # button (_on_mount_goto_click) re-enabled Start/Simulate on
    # "goto_result", which MountWorker emits IMMEDIATELY once :MS# is
    # ACCEPTED (code 0) -- well before the mount has actually arrived
    # (_poll_until_arrived keeps running for up to GOTO_POLL_TIMEOUT_S
    # afterward). That let an operator click Start/Simulate while the
    # real GOTO was still converging -- exactly the incident class
    # _disable_goto_buttons exists to prevent, just via the other GOTO
    # button. The real "done" signals are goto_arrived/goto_timeout.
    _select_a_pass(panel)
    panel.set_mount_connected(True)
    panel._on_arm_click()
    assert str(panel._start_button["state"]) == "normal"
    assert str(panel._simulate_button["state"]) == "normal"

    panel._mount_worker.goto = lambda *a, **kw: None
    panel._on_mount_goto_click()
    assert str(panel._mount_goto_button["state"]) == "disabled"
    assert str(panel._arm_button["state"]) == "disabled"
    assert str(panel._start_button["state"]) == "disabled"
    assert str(panel._simulate_button["state"]) == "disabled"

    # code=0 -- accepted, now slewing -- must NOT re-enable yet.
    panel.handle_mount_event(WorkerEvent("goto_result", {"code": 0, "meaning": "slewing"}))
    assert str(panel._mount_goto_button["state"]) == "disabled"
    assert str(panel._start_button["state"]) == "disabled"

    # The real completion signal.
    panel.handle_mount_event(WorkerEvent("goto_arrived", {"ra_hours": 1.0, "dec_deg": 45.0}))
    assert str(panel._mount_goto_button["state"]) == "normal"
    assert str(panel._arm_button["state"]) == "normal"
    assert str(panel._simulate_button["state"]) == "normal"
    assert str(panel._start_button["state"]) == "normal"  # was armed before the GOTO


def test_mount_goto_rejection_re_enables_immediately_no_polling_happens(panel):
    # code != 0 (below horizon, altitude limit, e7 not-synced, ...) means
    # the mount refused the target outright -- MountWorker never starts
    # polling in that case, so goto_result IS the final word here and
    # must re-enable right away, not wait for a goto_arrived/goto_timeout
    # that will never come.
    _select_a_pass(panel)
    panel.set_mount_connected(True)
    panel._mount_worker.goto = lambda *a, **kw: None
    panel._on_mount_goto_click()
    assert str(panel._mount_goto_button["state"]) == "disabled"

    panel.handle_mount_event(WorkerEvent("goto_result", {"code": 1, "meaning": "below horizon"}))
    assert str(panel._mount_goto_button["state"]) == "normal"
    assert str(panel._arm_button["state"]) == "normal"
    assert str(panel._simulate_button["state"]) == "normal"


def test_mount_goto_timeout_also_re_enables(panel):
    # goto_timeout: the mount accepted the target but never settled
    # within GOTO_POLL_TIMEOUT_S -- still a terminal state, must not
    # leave the operator stuck with every GOTO/Start/Simulate button
    # disabled forever.
    _select_a_pass(panel)
    panel.set_mount_connected(True)
    panel._mount_worker.goto = lambda *a, **kw: None
    panel._on_mount_goto_click()
    panel.handle_mount_event(WorkerEvent("goto_result", {"code": 0, "meaning": "slewing"}))
    assert str(panel._mount_goto_button["state"]) == "disabled"

    panel.handle_mount_event(WorkerEvent("goto_timeout", {"timeout_s": 15.0}))
    assert str(panel._mount_goto_button["state"]) == "normal"
    assert str(panel._arm_button["state"]) == "normal"
    assert str(panel._simulate_button["state"]) == "normal"


def test_training_error_checkbox_enables_only_when_the_connected_mount_is_mock(panel):
    assert str(panel._training_error_check["state"]) == "disabled"

    panel.handle_mount_event(WorkerEvent("connected", {"firmware": "1.0", "connection_kind": "mock"}))
    assert str(panel._training_error_check["state"]) == "normal"
    assert panel._mount_is_mock is True

    panel.handle_mount_event(WorkerEvent("disconnected", {}))
    assert str(panel._training_error_check["state"]) == "disabled"
    assert panel._mount_is_mock is False

    panel.handle_mount_event(WorkerEvent("connected", {"firmware": "1.0", "connection_kind": "serial"}))
    assert str(panel._training_error_check["state"]) == "disabled"
    assert panel._mount_is_mock is False


def test_training_error_checkbox_unchecks_itself_when_a_real_mount_connects(panel):
    # If it were left checked while disabled, the next mock session would
    # silently start with an injected error the operator never asked for.
    panel.handle_mount_event(WorkerEvent("connected", {"firmware": "1.0", "connection_kind": "mock"}))
    panel._training_error_var.set(True)

    panel.handle_mount_event(WorkerEvent("connected", {"firmware": "1.0", "connection_kind": "serial"}))

    assert panel._training_error_var.get() is False


def test_simulate_click_injects_a_training_pointing_error_when_checked_on_mock(panel):
    _select_a_pass(panel)
    panel.set_mount_connected(True)
    panel.handle_mount_event(WorkerEvent("connected", {"firmware": "1.0", "connection_kind": "mock"}))
    panel._training_error_var.set(True)

    injected = {}
    panel._mount_worker.inject_training_pointing_error = lambda ra, dec: injected.update(ra=ra, dec=dec)
    panel._mount_worker.start_tracking = lambda *a, **kw: None

    panel._on_simulate_click()

    assert "ra" in injected and "dec" in injected
    assert injected["ra"] != 0.0 or injected["dec"] != 0.0


def test_simulate_click_does_not_inject_when_checkbox_unchecked(panel):
    _select_a_pass(panel)
    panel.set_mount_connected(True)
    panel.handle_mount_event(WorkerEvent("connected", {"firmware": "1.0", "connection_kind": "mock"}))
    assert panel._training_error_var.get() is False  # default

    called = []
    panel._mount_worker.inject_training_pointing_error = lambda *a, **kw: called.append((a, kw))
    panel._mount_worker.start_tracking = lambda *a, **kw: None

    panel._on_simulate_click()

    assert called == []


def test_simulate_click_does_not_inject_when_mount_is_not_mock_even_if_checked(panel):
    # Defense in depth: _on_simulate_click checks _mount_is_mock itself,
    # not just the checkbox's enabled state -- belt and suspenders with
    # MountWorker's own refusal (see _handle_inject_training_pointing_error).
    _select_a_pass(panel)
    panel.set_mount_connected(True)
    panel._training_error_var.set(True)  # simulate a stale/forced-True var
    assert panel._mount_is_mock is False  # never told it's mock

    called = []
    panel._mount_worker.inject_training_pointing_error = lambda *a, **kw: called.append((a, kw))
    panel._mount_worker.start_tracking = lambda *a, **kw: None

    panel._on_simulate_click()

    assert called == []


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


@pytest.fixture
def alignment_panel():
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    site_vars = SiteVars.create()
    site_vars.lat.set("46.18")
    site_vars.lon.set("6.14")
    p = AlignmentPanel(root, mount_worker, AxisSigns(ra=1.0, dec=1.0), site_vars, finder_state=FinderState())
    yield p
    mount_worker.shutdown()
    root.destroy()


@pytest.fixture
def alignment_panel_factory():
    created = []

    def make(**kwargs):
        root = tk.Tk()
        root.withdraw()
        mount_worker = MountWorker()
        site_vars = SiteVars.create()
        site_vars.lat.set("46.18")
        site_vars.lon.set("6.14")
        p = AlignmentPanel(
            root, mount_worker, AxisSigns(ra=1.0, dec=1.0), site_vars, finder_state=FinderState(), **kwargs,
        )
        created.append((p, mount_worker, root))
        return p

    yield make
    for p, mount_worker, root in created:
        mount_worker.shutdown()
        root.destroy()


class _MplEvent:
    def __init__(self, xdata, ydata, button=None, inaxes=True):
        self.xdata = xdata
        self.ydata = ydata
        self.button = button
        self.inaxes = inaxes


@pytest.fixture
def sky_map():
    root = tk.Tk()
    root.withdraw()
    selected = []
    w = AlignmentSkyMapWidget(root, on_star_selected=lambda star: selected.append(star))
    w.selected = selected  # stash for tests to read
    yield w
    w.close()
    root.destroy()


def test_sky_map_scroll_zoom_is_centered_on_the_cursor_not_the_view_center(sky_map):
    w = sky_map
    w.ax.set_xlim(-90, 90)
    w.ax.set_ylim(-90, 90)
    # Scroll "up" (zoom in) with the cursor sitting off-center, near the
    # edge of the view -- a naive concentric zoom (the old polar-axes
    # implementation) would always shrink back toward (0, 0) regardless
    # of where the cursor was, which is exactly the bug being fixed here.
    cursor_x, cursor_y = 60.0, 60.0
    w._on_scroll(_MplEvent(xdata=cursor_x, ydata=cursor_y, button="up", inaxes=w.ax))
    new_xlim, new_ylim = w.ax.get_xlim(), w.ax.get_ylim()
    assert new_xlim[1] - new_xlim[0] < 180.0  # actually zoomed in
    # The cursor's data position should have stayed inside the new view
    # AND close to the same fractional position within it (not re-centered).
    assert new_xlim[0] < cursor_x < new_xlim[1]
    assert new_ylim[0] < cursor_y < new_ylim[1]
    old_fx, old_fy = (cursor_x - (-90.0)) / 180.0, (cursor_y - (-90.0)) / 180.0
    new_fx = (cursor_x - new_xlim[0]) / (new_xlim[1] - new_xlim[0])
    new_fy = (cursor_y - new_ylim[0]) / (new_ylim[1] - new_ylim[0])
    assert new_fx == pytest.approx(old_fx, abs=0.05)
    assert new_fy == pytest.approx(old_fy, abs=0.05)


def test_sky_map_zoom_out_never_exceeds_the_full_sky_extent(sky_map):
    w = sky_map
    for _ in range(10):
        w._on_scroll(_MplEvent(xdata=0.0, ydata=0.0, button="down", inaxes=w.ax))
    x0, x1 = w.ax.get_xlim()
    y0, y1 = w.ax.get_ylim()
    assert x0 >= -90.0 - 1e-6 and x1 <= 90.0 + 1e-6
    assert y0 >= -90.0 - 1e-6 and y1 <= 90.0 + 1e-6


def test_sky_map_click_selects_the_nearest_star(sky_map):
    from am5.named_stars import NAMED_STARS_BY_NAME
    w = sky_map
    sirius = NAMED_STARS_BY_NAME["Sirius"]
    vega = NAMED_STARS_BY_NAME["Vega"]
    w.set_stars([(sirius, 90.0, 40.0), (vega, 270.0, 60.0)])
    from am5.gui.panels import _altaz_to_xy
    x, y = _altaz_to_xy(90.0, 40.0)
    w._on_click(_MplEvent(xdata=x, ydata=y, inaxes=w.ax))
    assert w.selected == [sirius]


def test_sky_map_click_far_from_any_star_selects_nothing(sky_map):
    from am5.named_stars import NAMED_STARS_BY_NAME
    w = sky_map
    sirius = NAMED_STARS_BY_NAME["Sirius"]
    w.set_stars([(sirius, 90.0, 40.0)])
    w._on_click(_MplEvent(xdata=-80.0, ydata=-80.0, inaxes=w.ax))
    assert w.selected == []


def test_sky_map_update_mount_marker_does_not_raise(sky_map):
    sky_map.update_mount_marker(az_deg=180.0, alt_deg=45.0)
    assert sky_map._mount_marker is not None
    sky_map.update_mount_marker(az_deg=190.0, alt_deg=50.0)  # move, not recreate


def test_alignment_panel_position_event_updates_label_and_marker(alignment_panel):
    p = alignment_panel
    p.handle_mount_event(WorkerEvent("position", {"ra_hours": 5.5, "dec_deg": 45.0}))
    assert "5.500h" in p._mount_position_var.get()
    assert "45.00" in p._mount_position_var.get()
    assert p._sky_map._mount_marker is not None


def test_visible_named_stars_only_returns_stars_above_the_horizon():
    from datetime import datetime, timezone
    when = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    stars = visible_named_stars(46.18, 6.14, when, min_alt_deg=10.0)
    assert stars  # some named star should be up at some point on a given night
    for _star, _az, alt in stars:
        assert alt >= 10.0


def test_alignment_panel_selecting_a_star_enables_goto_and_sync(alignment_panel):
    p = alignment_panel
    assert str(p._goto_button["state"]) == "disabled"
    assert str(p._sync_button["state"]) == "disabled"
    p.set_connected(True)
    assert p._sky_map._stars  # some star should be visible right now at this real site/time
    star, _az, _alt = p._sky_map._stars[0]

    p._on_star_selected(star)
    assert p._selected_star is star
    assert star.name in p._selected_var.get()
    assert str(p._goto_button["state"]) == "normal"
    assert str(p._sync_button["state"]) == "normal"


def test_alignment_panel_goto_click_uses_native_mount_goto_not_jog_goto(alignment_panel):
    # Regression: this button used jog_goto, whose own divergence guard is
    # documented (am5.angles.angular_separation_deg's docstring) as meant
    # for short, close-in final-approach corrections -- not an arbitrary-
    # distance slew to a freshly-selected star anywhere on the sky map,
    # exactly this button's use case. Native goto (:MS#, firmware-driven
    # pier side) is the right tool here, same as TransitPanel's own
    # "GOTO (mount, auto pier side)" button.
    p = alignment_panel
    p.set_connected(True)
    star, _az, _alt = p._sky_map._stars[0]
    p._on_star_selected(star)

    called = []
    p._mount_worker.goto = lambda ra, dec: called.append((ra, dec))
    p._mount_worker.jog_goto = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not use jog_goto"))

    p._on_goto_click()

    assert called == [(star.ra_hours, star.dec_deg)]


def test_alignment_panel_goto_stays_disabled_until_arrived_not_just_accepted(alignment_panel):
    # Regression: goto_result's code==0 means ACCEPTED and slewing, not
    # arrived -- _poll_until_arrived keeps running well after this event
    # fires. Re-enabling here unconditionally would let the operator
    # click GOTO again while the mount is still converging on the first
    # target (same incident class as TransitPanel's own goto_result fix).
    p = alignment_panel
    p.set_connected(True)
    star, _az, _alt = p._sky_map._stars[0]
    p._on_star_selected(star)
    p._mount_worker.goto = lambda *a, **kw: None

    p._on_goto_click()
    assert str(p._goto_button["state"]) == "disabled"

    p.handle_mount_event(WorkerEvent("goto_result", {"code": 0, "meaning": "slewing"}))
    assert str(p._goto_button["state"]) == "disabled"

    p.handle_mount_event(WorkerEvent("goto_arrived", {"ra_hours": star.ra_hours, "dec_deg": star.dec_deg}))
    assert str(p._goto_button["state"]) == "normal"


def test_alignment_panel_goto_rejection_re_enables_immediately(alignment_panel):
    p = alignment_panel
    p.set_connected(True)
    star, _az, _alt = p._sky_map._stars[0]
    p._on_star_selected(star)
    p._mount_worker.goto = lambda *a, **kw: None

    p._on_goto_click()
    p.handle_mount_event(WorkerEvent("goto_result", {"code": 1, "meaning": "below horizon"}))

    assert str(p._goto_button["state"]) == "normal"


def test_alignment_panel_selecting_a_different_star_mid_goto_does_not_re_enable(alignment_panel):
    # Regression: _on_star_selected calls _refresh_widget_states() on
    # every sky-map click -- before _goto_in_progress existed, that
    # unconditionally re-derived the button's state from connected/
    # parked/has_star alone, so picking a different star while a GOTO was
    # still slewing re-enabled the button and let a second, overlapping
    # GOTO fire on top of the first.
    p = alignment_panel
    p.set_connected(True)
    star_a, _az, _alt = p._sky_map._stars[0]
    p._on_star_selected(star_a)
    p._mount_worker.goto = lambda *a, **kw: None
    p._on_goto_click()
    assert str(p._goto_button["state"]) == "disabled"

    star_b = p._sky_map._stars[1][0] if len(p._sky_map._stars) > 1 else star_a
    p._on_star_selected(star_b)

    assert str(p._goto_button["state"]) == "disabled"


def test_alignment_panel_disconnect_mid_goto_resets_stuck_in_progress_flag(alignment_panel):
    # Regression: same class of bug as CalibrationPanel's own _calib_step
    # disconnect reset -- a disconnect mid-GOTO left _goto_in_progress
    # stuck True forever, leaving the button disabled even after
    # reconnecting, with no recovery short of restarting the app.
    p = alignment_panel
    p.set_connected(True)
    star, _az, _alt = p._sky_map._stars[0]
    p._on_star_selected(star)
    p._mount_worker.goto = lambda *a, **kw: None
    p._on_goto_click()
    assert p._goto_in_progress is True

    p.set_connected(False)
    assert p._goto_in_progress is False

    p.set_connected(True)
    p._on_star_selected(star)
    assert str(p._goto_button["state"]) == "normal"


def test_alignment_panel_sync_button_ignores_disconnected_state_for_sync_only(alignment_panel):
    # Sync never moves the mount -- only needs a connection, not an
    # unparked state (mirrors JogWindow's own sync button).
    p = alignment_panel
    p.set_connected(True)
    star, _az, _alt = p._sky_map._stars[0]
    p._on_star_selected(star)
    p.handle_mount_event(WorkerEvent("parked", {}))
    assert str(p._goto_button["state"]) == "disabled"  # GOTO IS blocked while parked
    assert str(p._sync_button["state"]) == "normal"


def test_alignment_panel_sync_click_queues_worker_sync_and_updates_status(alignment_panel):
    p = alignment_panel
    p.set_connected(True)
    star, _az, _alt = p._sky_map._stars[0]
    p._on_star_selected(star)
    p._on_sync_click()
    assert str(p._sync_button["state"]) == "disabled"
    assert star.name in p._status_var.get()

    p.handle_mount_event(WorkerEvent("sync_result", {
        "ok": True, "message": "Synced", "ra_hours": star.ra_hours, "dec_deg": star.dec_deg,
    }))
    assert p._status_var.get() == "Synced"


def test_alignment_panel_mode_toggle_updates_status_from_worker_event(alignment_panel):
    p = alignment_panel
    p.set_connected(True)
    p.handle_mount_event(WorkerEvent("alignment_status", {"enabled": True, "point_count": 3}))
    assert p._alignment_mode is True
    assert p._alignment_mode_var.get() is True
    assert "3 point" in p._alignment_status_var.get()

    p.handle_mount_event(WorkerEvent("alignment_status", {"enabled": False, "point_count": 0}))
    assert p._alignment_mode is False
    assert "Off" in p._alignment_status_var.get()


def test_alignment_panel_sync_requests_a_status_refresh_while_in_alignment_mode(alignment_panel):
    p = alignment_panel
    p.set_connected(True)
    star, _az, _alt = p._sky_map._stars[0]
    p._on_star_selected(star)
    p._alignment_mode = True  # simulate the worker having already confirmed alignment mode is on

    calls = []
    p._mount_worker.read_alignment_status = lambda: calls.append(True)
    p.handle_mount_event(WorkerEvent("sync_result", {
        "ok": True, "message": "Synced", "ra_hours": star.ra_hours, "dec_deg": star.dec_deg,
    }))
    assert calls == [True]


def test_alignment_panel_turning_off_alignment_mode_asks_for_confirmation(alignment_panel, monkeypatch):
    p = alignment_panel
    p.set_connected(True)
    p._alignment_mode = True  # was on
    p._alignment_mode_var.set(False)  # operator just unchecked it

    from am5.gui import panels as panels_module
    calls = []
    monkeypatch.setattr(panels_module.messagebox, "askyesno", lambda *a, **k: (calls.append(1), False)[1])
    p._on_alignment_mode_toggle()
    assert calls == [1]
    assert p._alignment_mode_var.get() is True  # declined -- reverted the checkbox


class _FakeSolveResult:
    def __init__(
        self, ra_deg: float, dec_deg: float, success: bool = True, message: str = "",
        pixel_scale_arcsec: float = 1.72, field_rotation_deg: float = 0.0, flip_parity: bool = False,
    ):
        self.success = success
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.message = message
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self.field_rotation_deg = field_rotation_deg
        self.flip_parity = flip_parity


def test_polar_alignment_reads_frame_and_scale_from_the_selected_camera(alignment_panel):
    p = alignment_panel
    p._finder_state.last_frame = np.zeros((10, 10), dtype=np.uint8)
    p._finder_state.finder_plate_scale_arcsec = 5.0
    p._finder_state.last_main_frame = np.ones((20, 20), dtype=np.uint8)
    p._finder_state.main_plate_scale_arcsec = 2.0

    p._polar_camera_var.set("finder")
    frame, scale = p._current_frame_and_plate_scale()
    assert frame.shape == (10, 10)
    assert scale == 5.0

    p._polar_camera_var.set("main")
    frame, scale = p._current_frame_and_plate_scale()
    assert frame.shape == (20, 20)
    assert scale == 2.0


def test_polar_alignment_refuses_to_start_without_a_frame(alignment_panel):
    p = alignment_panel
    p.set_connected(True)
    p._finder_state.last_frame = None
    p._on_polar_start_click()
    assert "connected" in p._polar_status_var.get()
    assert str(p._polar_start_button["state"]) == "normal"


def test_polar_alignment_rejects_invalid_rotation_or_rate(alignment_panel):
    p = alignment_panel
    p._polar_rotation_deg_var.set("not-a-number")
    p._on_polar_start_click()
    assert "Invalid" in p._polar_status_var.get()

    p._polar_rotation_deg_var.set("30")
    p._polar_rate_var.set("-5")
    p._on_polar_start_click()
    assert "positive" in p._polar_status_var.get()

    p._polar_rotation_deg_var.set("0")
    p._polar_rate_var.set("150")
    p._on_polar_start_click()
    assert "nonzero" in p._polar_status_var.get()


def test_polar_alignment_negative_rotation_jogs_west(alignment_panel, monkeypatch):
    p = alignment_panel
    p.set_connected(True)
    p._finder_state.last_frame = np.zeros((10, 10), dtype=np.uint8)
    p._finder_state.finder_plate_scale_arcsec = 5.0
    monkeypatch.setattr(p, "after", lambda ms, cb: cb())
    monkeypatch.setattr(
        p._solvers[p._solver_engine_var.get()], "solve_async",
        lambda frame, widget, on_done, **kw: on_done(_FakeSolveResult(0.0, 89.0)),
    )
    jog_calls = []
    monkeypatch.setattr(p._mount_worker, "jog_start", lambda direction, rate_x: jog_calls.append(("start", direction)))
    monkeypatch.setattr(p._mount_worker, "jog_stop", lambda direction: jog_calls.append(("stop", direction)))

    p._polar_rotation_deg_var.set("-30")
    p._polar_rate_var.set("150")
    p._on_polar_start_click()

    assert ("start", "w") in jog_calls
    assert ("stop", "w") in jog_calls
    assert ("start", "e") not in jog_calls
    assert p._polar_rotation_deg == pytest.approx(30.0)  # stored as a positive magnitude


def test_polar_alignment_aborts_after_all_retry_attempts_fail(alignment_panel, monkeypatch):
    # Regression: a single failed solve used to abort the whole 3-point
    # sequence immediately, forcing the operator to redo all 3 points
    # from scratch -- e.g. because the frame right after the mount
    # stopped rotating between points still showed real motion blur.
    # Now retries POLAR_SOLVE_RETRY_ATTEMPTS times (each re-reading the
    # live frame fresh) before actually giving up.
    p = alignment_panel
    p.set_connected(True)
    p._finder_state.last_frame = np.zeros((10, 10), dtype=np.uint8)
    p._finder_state.finder_plate_scale_arcsec = 5.0
    call_count = 0

    def fake_solve_async(frame, widget, on_done, **kw):
        nonlocal call_count
        call_count += 1
        on_done(_FakeSolveResult(0.0, 0.0, success=False, message="no stars found"))

    monkeypatch.setattr(p._solvers[p._solver_engine_var.get()], "solve_async", fake_solve_async)
    p._on_polar_start_click()

    assert call_count == POLAR_SOLVE_RETRY_ATTEMPTS
    assert "failed" in p._polar_status_var.get()
    assert "no stars found" in p._polar_status_var.get()
    assert str(p._polar_start_button["state"]) == "normal"


def test_polar_alignment_saves_every_failed_attempts_frame_when_out_dir_is_set(alignment_panel_factory, tmp_path, monkeypatch):
    # Regression: a failed solve attempt used to just discard its frame --
    # nothing to inspect afterward if a point failed all its retries (as
    # happened during real-hardware testing). Now each failed attempt's
    # exact frame is saved as FITS under out_dir/paa_failed_solves/<run>/.
    p = alignment_panel_factory(out_dir=tmp_path)
    p.set_connected(True)
    p._finder_state.last_frame = np.full((10, 10), 7, dtype=np.uint8)
    p._finder_state.finder_plate_scale_arcsec = 5.0

    def fake_solve_async(frame, widget, on_done, **kw):
        on_done(_FakeSolveResult(0.0, 0.0, success=False, message="no stars found"))

    monkeypatch.setattr(p._solvers[p._solver_engine_var.get()], "solve_async", fake_solve_async)
    p._on_polar_start_click()

    run_dir = tmp_path / "paa_failed_solves" / p._polar_run_label
    saved = sorted(run_dir.glob("point1_attempt*.fits"))
    assert len(saved) == POLAR_SOLVE_RETRY_ATTEMPTS
    from astropy.io import fits
    data = fits.getdata(saved[0])
    np.testing.assert_array_equal(data, p._finder_state.last_frame)
    assert str(run_dir) in p._polar_status_var.get()


def test_polar_alignment_preview_applies_the_shared_camera_stretch(alignment_panel_factory, monkeypatch):
    # Regression: _refresh_polar_preview never passed the per-camera
    # manual stretch through to show_frame_on_canvas, unlike every other
    # live preview (TransitPanel/FinderCameraPanel/FinderWindow/
    # CalibrationPanel) -- a gain change was invisible here even after
    # the stretch feature was added everywhere else.
    p = alignment_panel_factory()
    p._finder_state.last_frame = np.full((10, 10), 7, dtype=np.uint8)
    p._finder_state.finder_plate_scale_arcsec = 5.0
    p._finder_camera_vars.stretch.set(3.5)

    seen = {}

    def fake_show_frame_on_canvas(canvas, frame, stretch=1.0):
        seen["stretch"] = stretch
        return None

    monkeypatch.setattr(gui_panels, "show_frame_on_canvas", fake_show_frame_on_canvas)
    p._refresh_polar_preview()

    assert seen["stretch"] == 3.5


def test_polar_alignment_settle_delay_is_configurable_and_used_between_rotation_and_capture(alignment_panel, monkeypatch):
    # Regression: the settle delay between stopping the rotation and
    # capturing the next point used to be a hardcoded 800ms -- reported
    # live that the very first capture right after a rotation sometimes
    # still shows motion blur. Now an operator-adjustable field
    # (_polar_settle_s_var), read into self._polar_settle_s at run start
    # and used as the actual self.after() delay in _stop_polar_rotation.
    p = alignment_panel
    p.set_connected(True)
    p._finder_state.last_frame = np.zeros((10, 10), dtype=np.uint8)
    p._finder_state.finder_plate_scale_arcsec = 5.0
    monkeypatch.setattr(p._mount_worker, "jog_start", lambda *a, **kw: None)
    monkeypatch.setattr(p._mount_worker, "jog_stop", lambda *a, **kw: None)
    monkeypatch.setattr(
        p._solvers[p._solver_engine_var.get()], "solve_async",
        lambda frame, widget, on_done, **kw: on_done(_FakeSolveResult(0.0, 85.0)),
    )
    after_calls = []
    real_after = p.after

    def fake_after(ms, cb):
        after_calls.append(ms)
        return real_after(0, cb)

    monkeypatch.setattr(p, "after", fake_after)

    p._polar_rotation_deg_var.set("30")
    p._polar_rate_var.set("150")
    p._polar_settle_s_var.set("2.5")
    p._on_polar_start_click()
    p.update()

    assert 2500 in after_calls


def test_polar_alignment_retries_on_a_fresh_frame_until_it_succeeds(alignment_panel, monkeypatch):
    p = alignment_panel
    p.set_connected(True)
    p._finder_state.last_frame = np.zeros((10, 10), dtype=np.uint8)
    p._finder_state.finder_plate_scale_arcsec = 5.0
    monkeypatch.setattr(p, "after", lambda ms, cb: cb())
    monkeypatch.setattr(p._mount_worker, "jog_start", lambda *a, **kw: None)
    monkeypatch.setattr(p._mount_worker, "jog_stop", lambda *a, **kw: None)
    call_count = 0

    def fake_solve_async(frame, widget, on_done, **kw):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            on_done(_FakeSolveResult(0.0, 0.0, success=False, message="no stars found"))
        else:
            on_done(_FakeSolveResult(10.0, 20.0))

    monkeypatch.setattr(p._solvers[p._solver_engine_var.get()], "solve_async", fake_solve_async)
    p._on_polar_start_click()

    # Point 1 succeeded on its 3rd attempt -- the sequence must have kept
    # going (rotated + captured point 2) rather than aborting.
    assert call_count > 3
    assert "failed" not in p._polar_status_var.get()


def test_polar_alignment_full_workflow_computes_a_result(alignment_panel, monkeypatch):
    p = alignment_panel
    p.set_connected(True)
    p._finder_state.last_frame = np.zeros((10, 10), dtype=np.uint8)
    p._finder_state.finder_plate_scale_arcsec = 5.0

    # Collapse the jog-rotation timer waits into immediate execution --
    # the real defaults would otherwise make this test wait tens of
    # real seconds for a scheduled self.after() callback.
    monkeypatch.setattr(p, "after", lambda ms, cb: cb())

    # 3 synthetic solves tracing a circle around the true celestial pole
    # (same construction as tests/test_polar_alignment.py's own known-
    # axis test) -- a perfectly-aligned mount, so the reported error
    # should come out at ~0.
    solved_points = iter([(0.0, 85.0), (120.0, 85.0), (240.0, 85.0)])
    monkeypatch.setattr(
        p._solvers[p._solver_engine_var.get()], "solve_async",
        lambda frame, widget, on_done, **kw: on_done(_FakeSolveResult(*next(solved_points))),
    )
    jog_calls = []
    monkeypatch.setattr(p._mount_worker, "jog_start", lambda direction, rate_x: jog_calls.append(("start", direction, rate_x)))
    monkeypatch.setattr(p._mount_worker, "jog_stop", lambda direction: jog_calls.append(("stop", direction)))

    p._polar_rotation_deg_var.set("30")
    p._polar_rate_var.set("150")
    p._on_polar_start_click()

    assert "Total error" in p._polar_result_var.get()
    assert jog_calls.count(("start", "e", 150.0)) == 2  # rotated between each of the 3 captures
    assert jog_calls.count(("stop", "e")) == 2
    assert str(p._polar_start_button["state"]) == "normal"

    # The fitted axis's live-view locator dot (delta_col, delta_row pixel
    # offset) is populated once the measurement completes.
    assert p._polar_overlay is not None
    delta_col, delta_row = p._polar_overlay
    assert isinstance(delta_col, float)
    assert isinstance(delta_row, float)
    assert p._polar_last_alignment_result is not None


def test_polar_alignment_draws_two_fixed_direction_correction_arrows(alignment_panel, monkeypatch):
    # Regression: the live-view overlay used to project the correction
    # into the actual image's pixel space (a to-scale line from the
    # fitted axis toward the true pole) -- for any real (degree-scale)
    # error, that target is normally well outside the finder's <2deg
    # field of view, so the line was mostly invisible/off-canvas ("part
    # n'importe où", reported live). Fixed-length arrows in image space
    # were tried next, but real-hardware testing showed an operator can't
    # reliably map "arrow points toward the top-left of this star field"
    # to "turn this physical adjuster this way" -- reported live as a
    # real mis-adjustment (altitude moved the wrong way). Redesigned as a
    # KStars-PAA-style indicator with two arrows in FIXED screen
    # directions (up/down for altitude, left/right for azimuth),
    # independent of the camera's image orientation entirely -- see
    # _draw_polar_correction_arrows' own docstring.
    p = alignment_panel
    p.set_connected(True)
    p._finder_state.last_frame = np.zeros((10, 10), dtype=np.uint8)
    p._finder_state.finder_plate_scale_arcsec = 5.0
    monkeypatch.setattr(p, "after", lambda ms, cb: cb())

    # 3 synthetic solves tracing a circle around a KNOWN, deliberately
    # off-true-pole axis (47, 88) -- same rotation-matrix construction as
    # tests/test_polar_alignment.py's own test_fit_rotation_axis_recovers_
    # a_known_axis (3 equally-spaced RA points at the same declination
    # always fit back to the true pole exactly, so that shape alone can't
    # produce a nonzero error to test against; this rotates that trivial
    # case onto a real off-pole axis instead).
    def _unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
        ra, dec = math.radians(ra_deg), math.radians(dec_deg)
        return np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])

    def _to_radec(v: np.ndarray) -> tuple[float, float]:
        v = v / np.linalg.norm(v)
        dec_deg = math.degrees(math.asin(max(-1.0, min(1.0, v[2]))))
        ra_deg = math.degrees(math.atan2(v[1], v[0])) % 360.0
        return ra_deg, dec_deg

    def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)
        theta = math.radians(angle_deg)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)

    true_axis_ra, true_axis_dec = 47.0, 88.0
    pole = np.array([0.0, 0.0, 1.0])
    target = _unit_vector(true_axis_ra, true_axis_dec)
    rot_axis = np.cross(pole, target)
    rot_angle_deg = math.degrees(math.acos(np.clip(np.dot(pole, target), -1.0, 1.0)))
    r = _rotation_matrix(rot_axis, rot_angle_deg)
    solved_points = iter(_to_radec(r @ _unit_vector(phase, 85.0)) for phase in (0.0, 120.0, 240.0))
    monkeypatch.setattr(
        p._solvers[p._solver_engine_var.get()], "solve_async",
        lambda frame, widget, on_done, **kw: on_done(_FakeSolveResult(*next(solved_points))),
    )
    monkeypatch.setattr(p._mount_worker, "jog_start", lambda *a, **kw: None)
    monkeypatch.setattr(p._mount_worker, "jog_stop", lambda *a, **kw: None)
    p._polar_rotation_deg_var.set("30")
    p._polar_rate_var.set("150")
    p._on_polar_start_click()

    assert p._polar_last_alignment_result is not None
    assert abs(p._polar_last_alignment_result.error_deg) > 1.0  # a real, several-degree error

    result = p._polar_last_alignment_result
    canvas = p._polar_correction_canvas
    items = canvas.find_all()
    # A shaft (line) + an arrowhead (polygon) + a text label per axis.
    line_items = [i for i in items if canvas.type(i) == "line"]
    polygon_items = [i for i in items if canvas.type(i) == "polygon"]
    text_items = [i for i in items if canvas.type(i) == "text"]
    assert len(line_items) == 2
    assert len(polygon_items) == 2
    assert len(text_items) == 2
    labels = " ".join(canvas.itemcget(i, "text") for i in text_items)
    assert "ALT" in labels
    assert "AZ" in labels
    alt_word = "lower" if result.error_alt_deg > 0 else "raise"
    az_word = "west" if result.error_az_deg > 0 else "east"
    assert alt_word in labels
    assert az_word in labels
    assert f"{abs(result.error_alt_deg) * 60.0:.1f}" in labels
    assert f"{abs(result.error_az_deg) * 60.0:.1f}" in labels

    # The whole point of the redesign: arrow direction is a FIXED screen
    # direction (up/down, left/right), never derived from image pixel
    # projection -- assert the altitude shaft is vertical and the azimuth
    # shaft is horizontal, regardless of what the (here, real off-pole)
    # solve's own field_rotation_deg/flip_parity happen to be.
    alt_line = [i for i in line_items if abs(canvas.coords(i)[0] - canvas.coords(i)[2]) < 0.01][0]
    az_line = [i for i in line_items if abs(canvas.coords(i)[1] - canvas.coords(i)[3]) < 0.01][0]
    assert alt_line != az_line


def test_polar_alignment_overlay_is_cleared_at_the_start_of_a_new_run(alignment_panel, monkeypatch):
    p = alignment_panel
    p.set_connected(True)
    p._finder_state.last_frame = np.zeros((10, 10), dtype=np.uint8)
    p._finder_state.finder_plate_scale_arcsec = 5.0
    p._polar_overlay = (1.0, 2.0)  # stale, from a previous run

    monkeypatch.setattr(
        p._solvers[p._solver_engine_var.get()], "solve_async",
        lambda frame, widget, on_done, **kw: on_done(_FakeSolveResult(0.0, 0.0, success=False, message="no stars")),
    )
    p._polar_rotation_deg_var.set("30")
    p._polar_rate_var.set("150")
    p._on_polar_start_click()

    assert p._polar_overlay is None


def _pass_trajectory(n: int = 20) -> Trajectory:
    # A simple rise-to-set arc: azimuth sweeps 90->270 while altitude rises
    # then falls back to the horizon -- enough for set_pass_track to have
    # a real multi-point line to draw, not just a single point.
    t_unix = time.time() + np.linspace(0, 300, n)
    az_deg = np.linspace(90.0, 270.0, n)
    alt_deg = 60.0 * np.sin(np.linspace(0, np.pi, n))
    return Trajectory(
        t_unix=t_unix, ra_deg=np.full(n, 45.0), dec_deg=np.full(n, 45.0),
        dra_dt_deg_s=np.full(n, 0.01), ddec_dt_deg_s=np.full(n, 0.005),
        alt_deg=alt_deg, az_deg=az_deg, ha_hours=np.zeros(n),
        distance_km=np.full(n, 500.0),
    )


def test_alignment_panel_set_trajectory_updates_status_and_forwards_to_sky_map(alignment_panel, monkeypatch):
    p = alignment_panel
    calls = []
    monkeypatch.setattr(p._sky_map, "set_pass_track", lambda az, alt: calls.append((list(az), list(alt))))
    trajectory = _pass_trajectory()
    window = _window(datetime(2026, 7, 16, 20, 0, 0, tzinfo=timezone.utc), duration_s=300.0, max_elevation_deg=60.0)

    p.set_trajectory(trajectory, window, "ISS (ZARYA)")

    assert len(calls) == 1
    assert calls[0][0] == list(trajectory.az_deg)
    assert calls[0][1] == list(trajectory.alt_deg)
    assert "ISS (ZARYA)" in p._pass_track_var.get()
    assert "20:00:00" in p._pass_track_var.get()
    assert "60" in p._pass_track_var.get()


def test_sky_map_set_pass_track_draws_a_line_and_survives_set_stars(sky_map):
    w = sky_map
    trajectory = _pass_trajectory()
    w.set_pass_track(trajectory.az_deg, trajectory.alt_deg)
    assert w._pass_track_azalt is not None
    lines_before = list(w.ax.get_lines())
    assert len(lines_before) >= 1  # at least the track line itself

    # set_stars clears+redraws the axes (periodic star refresh) -- the
    # pass track must survive that, same as the star field/mount marker.
    w.set_stars([])
    lines_after = list(w.ax.get_lines())
    assert len(lines_after) >= 1


def _blob(x: float, y: float, found: bool = True) -> BlobDetection:
    return BlobDetection(found=found, centroid_x=x, centroid_y=y, peak_value=200.0, pixel_count=20)


def _make_camera_calibration_tab_visible(panel):
    """handle_camera_event/handle_finder_camera_event now skip building the
    preview PhotoImage while their pane isn't actually mapped (see the real
    gate's own rationale -- a ttk.Notebook only maps the selected page's
    widgets, and CalibrationPanel's preview panes live inside its nested
    "Camera calibration" sub-tab specifically). Tests that assert on the
    rendered preview need the panel actually packed, visible, and on the
    right sub-tab -- not the usual root.withdraw() headless setup every
    other test in this file relies on."""
    root = panel.winfo_toplevel()
    root.deiconify()
    panel.pack()
    panel._sub_notebook.select(1)  # "Camera calibration"
    root.update()


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


def test_calibration_panel_sibling_motion_buttons_disabled_during_guiding_calibration(calibration_panel):
    # Regression: _axis_calibrate_button and _lag_measure_button used to
    # stay clickable while the RA/DEC guiding-calibration nudge sequence
    # was actively jogging the mount -- both drive their own mount motion
    # with no guard against an in-progress jog (see _handle_calibrate/
    # _handle_measure_mount_lag in worker.py), so a click mid-nudge could
    # queue a second motion command that runs on the worker thread while
    # the guiding-calibration nudge is still physically moving the mount.
    p = calibration_panel
    p.set_connected(True)
    p._latest_radec = (3.0, 45.0)
    p._latest_blob = _blob(100.0, 100.0)

    p._on_calibrate_click()
    assert p._calib_step == "ra"
    assert str(p._calibrate_button["state"]) == "disabled"
    assert str(p._axis_calibrate_button["state"]) == "disabled"
    assert str(p._lag_measure_button["state"]) == "disabled"

    p._latest_radec = (3.0 + (10.0 / 3600.0) / 15.0, 45.0)  # +10 arcsec RA
    p._latest_blob = _blob(140.0, 100.0)
    p._calib_ra_measure()
    assert p._calib_step == "dec"
    assert str(p._axis_calibrate_button["state"]) == "disabled"
    assert str(p._lag_measure_button["state"]) == "disabled"

    p._latest_radec = (p._latest_radec[0], 45.0 + 8.0 / 3600.0)  # +8 arcsec DEC
    p._latest_blob = _blob(140.0, 130.0)
    p._calib_dec_measure()

    assert p._calib_step is None
    assert str(p._calibrate_button["state"]) == "normal"
    assert str(p._axis_calibrate_button["state"]) == "normal"
    assert str(p._lag_measure_button["state"]) == "normal"


def test_calibration_panel_disconnect_mid_calibration_resets_stuck_step(calibration_panel):
    # Regression: _calib_step was never reset on disconnect -- a disconnect
    # mid-sequence left it stuck at "ra"/"dec" forever, and _on_calibrate_
    # click's own re-entry guard (`if self._calib_step is not None: return`)
    # would then silently no-op every future click, even after reconnecting,
    # with no recovery short of restarting the app.
    p = calibration_panel
    p.set_connected(True)
    p._latest_radec = (3.0, 45.0)
    p._latest_blob = _blob(100.0, 100.0)
    p._on_calibrate_click()
    assert p._calib_step == "ra"

    p.set_connected(False)
    assert p._calib_step is None

    p.set_connected(True)
    p._latest_radec = (3.0, 45.0)
    p._latest_blob = _blob(100.0, 100.0)
    p._on_calibrate_click()
    assert p._calib_step == "ra"


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
    p._active_trajectory = _guiding_trajectory()

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
    p._active_trajectory = _guiding_trajectory()

    p._maybe_apply_auto_guide_correction((480, 640), _blob(640 / 2 + 1, 480 / 2))
    _, perp = p._live_offsets.snapshot()
    assert perp == 0.0


def test_calibration_panel_auto_guide_correction_points_toward_the_target_not_away(calibration_panel):
    # Regression: _maybe_apply_auto_guide_correction used to feed
    # decompose_error's raw (actual-target) cross_deg straight into
    # trigger_perp_pulse's sign. calibrate_from_nudges builds the
    # calibration from the BLOB's own pixel shift when the MOUNT is
    # nudged by a known sky amount (target held fixed) -- nudging the
    # mount +d_ra moves the boresight TOWARD a target ahead of it, so the
    # blob's measured shift is the pixel image of -d_ra, not +d_ra. That
    # flips the matrix GuidingCalibration stores, so pixel_to_sky() on a
    # live frame comes out as (actual - target), not (target - actual) as
    # the old code assumed -- confirmed by hand and by feeding the real
    # calibrate_from_nudges/pixel_to_sky/decompose_error/
    # _perp_rate_components chain a known scenario: the un-negated version
    # pushed the commanded rate AWAY from the target, which would have
    # driven the ISS out of frame instead of centering it.
    p = calibration_panel
    # Calibrate against a known, simple optical mapping (1 arcsec == 1 px,
    # no rotation) exactly the way the real "Calibrate camera-to-sky
    # mapping" flow does: nudge mount by d_ra/d_dec, record the blob's own
    # resulting pixel shift (target fixed, so the blob moves opposite the
    # boresight's own motion).
    d_ra_nudge, d_dec_nudge = 10.0, 10.0
    p._calibration = calibrate_from_nudges(
        d_ra_nudge, -d_ra_nudge, 0.0,
        d_dec_nudge, 0.0, -d_dec_nudge,
    )

    # dec=0 with velocity purely along DEC makes the "cross" axis align
    # exactly with RA-tan, so the physically correct correction direction
    # is unambiguous: a blob 10px east of centre (under this calibration)
    # is the image of a target 10" AHEAD of the boresight in RA -- the
    # mount is lagging and needs a POSITIVE extra RA rate to catch up.
    n = 10
    t_unix = time.time() + np.linspace(-5, 5, n)
    p._active_trajectory = Trajectory(
        t_unix=t_unix, ra_deg=np.zeros(n), dec_deg=np.zeros(n),
        dra_dt_deg_s=np.zeros(n), ddec_dt_deg_s=np.full(n, 0.001),
        alt_deg=np.full(n, 45.0), az_deg=np.full(n, 180.0), ha_hours=np.zeros(n),
        distance_km=np.full(n, 500.0),
    )

    p._maybe_apply_auto_guide_correction((480, 640), _blob(640 / 2 + 10, 480 / 2))

    _, perp_sign = p._live_offsets.snapshot()
    assert perp_sign != 0.0

    extra_dra_dt, extra_ddec_dt = _perp_rate_components(0.0, 0.0, 0.001, perp_sign)
    assert extra_dra_dt > 0, (
        "auto-guide correction points away from the target instead of toward it "
        f"(extra_dra_dt={extra_dra_dt}, should be positive to catch up a lagging boresight)"
    )


def test_calibration_panel_auto_guide_skips_outside_the_trajectorys_active_window(calibration_panel):
    # Regression: outside the trajectory's own [t_unix[0], t_unix[-1]]
    # window (a pass selected in advance, or tracking started early and
    # still sitting at the boundary -- both explicitly supported, see
    # Trajectory.interpolate's own docstring), interpolate() zeroes dra_dt/
    # ddec_dt. decompose_error's zero-speed branch then returns a bare
    # magnitude for cross (always >= 0, no directional information --
    # confirmed by feeding it the same error with the sign flipped and
    # getting an identical result), which the correction's own negation
    # turns into an always-negative "correction" regardless of the TRUE
    # error direction -- a real hazard during an early start, since the
    # tracking loop IS already consuming LiveOffsets every tick while
    # sitting at the boundary.
    p = calibration_panel
    p._calibration = calibrate_from_nudges(10.0, -10.0, 0.0, 10.0, 0.0, -10.0)

    n = 10
    t_unix = time.time() + np.linspace(100.0, 200.0, n)  # starts 100s in the future -- "now" is before t_unix[0]
    p._active_trajectory = Trajectory(
        t_unix=t_unix, ra_deg=np.zeros(n), dec_deg=np.zeros(n),
        dra_dt_deg_s=np.zeros(n), ddec_dt_deg_s=np.full(n, 0.001),
        alt_deg=np.full(n, 45.0), az_deg=np.full(n, 180.0), ha_hours=np.zeros(n),
        distance_km=np.full(n, 500.0),
    )

    triggered = []
    p._live_offsets.trigger_perp_pulse = lambda sign, **kw: triggered.append(sign)

    p._maybe_apply_auto_guide_correction((480, 640), _blob(640 / 2 + 10, 480 / 2))

    assert triggered == []


def test_calibration_panel_handle_camera_event_updates_blob_and_preview(calibration_panel):
    p = calibration_panel
    _make_camera_calibration_tab_visible(p)
    frame = np.full((60, 80), 15, dtype=np.uint8)
    frame[25:35, 55:65] = 220  # a bright synthetic ISS blob
    p.handle_camera_event(CameraEvent(kind="preview_frame", payload={"pgm": frame_to_pgm(frame), "width": 80, "height": 60}))
    assert p._latest_blob is not None
    assert p._latest_blob.found is True
    assert p._preview_image is not None
    assert "ISS at pixel" in p._blob_status_var.get()


def test_calibration_panel_handle_finder_camera_event_renders_finder_preview():
    # The Camera calibration sub-tab shows both live feeds side by side
    # (see _build_camera_calibration_tab) so the operator can run either
    # calibration without switching tabs -- the finder side is fed via
    # handle_finder_camera_event (routed from App._pump_events alongside
    # FinderCameraPanel/FinderWindow's own handlers), not handle_camera_
    # event (that's the main camera's).
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    finder_state = FinderState()
    try:
        p = CalibrationPanel(root, mount_worker, camera_worker, LiveOffsets(), finder_state=finder_state)
        _make_camera_calibration_tab_visible(p)
        assert p._finder_preview_image is None

        frame = np.full((60, 80), 15, dtype=np.uint8)
        frame[25:35, 55:65] = 220
        p.handle_finder_camera_event(CameraEvent(kind="preview_frame", payload={"pgm": frame_to_pgm(frame), "width": 80, "height": 60}))

        assert p._finder_preview_image is not None
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_calibration_panel_finder_preview_draws_main_camera_fov_rectangle():
    # Same overlay as FinderCameraPanel's own preview canvas (main
    # camera's FOV rectangle via FinderState.main_fov_corners_px) --
    # drawn here too so an operator running "Calibrate fields" from this
    # tab can see where the main camera's field sits within the finder's
    # wider view without switching to the Finder tab.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    finder_state = FinderState()
    try:
        p = CalibrationPanel(root, mount_worker, camera_worker, LiveOffsets(), finder_state=finder_state)
        _make_camera_calibration_tab_visible(p)
        finder_state.calibration = FinderCalibration(
            calibrated=True, offset_row=0.0, offset_col=0.0, plate_scale_ratio=1.0, rotation_rad=0.0,
        )
        finder_state.main_sensor_width = 40
        finder_state.main_sensor_height = 40
        finder_state.last_frame = np.zeros((60, 80), dtype=np.uint8)  # main_fov_corners_px needs this for frame_shape

        frame = np.full((60, 80), 15, dtype=np.uint8)
        p.handle_finder_camera_event(CameraEvent(kind="preview_frame", payload={"pgm": frame_to_pgm(frame), "width": 80, "height": 60}))

        item_types = [p._finder_preview_canvas.type(item) for item in p._finder_preview_canvas.find_all()]
        assert "polygon" in item_types
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_calibration_panel_main_camera_disconnect_clears_stale_blob_lock(calibration_panel):
    # Regression: handle_camera_event had no "disconnected" branch at all
    # (the only one in the file without one) -- a main-camera disconnect
    # while auto-guide had a lock left FinderState.main_blob_locked stuck
    # True forever. TransitPanel reads it on every tracking_tick
    # completely independently of whether preview_frame events are still
    # arriving, so finder correction would stay silently locked out for
    # the rest of the session with no main camera connected at all.
    p = calibration_panel
    finder_state = FinderState()
    p._finder_state = finder_state
    finder_state.set_main_blob_locked(True)
    p._latest_blob = _blob(10.0, 10.0)
    p._blob_status_var.set("ISS at pixel (10, 10), peak 200")

    p.handle_camera_event(CameraEvent(kind="disconnected", payload={}))

    assert finder_state.main_blob_locked is False
    assert p._latest_blob is None
    assert p._blob_status_var.get() == "No frame yet"


def test_calibration_panel_blob_position_is_full_resolution_for_a_frame_above_the_preview_cap(calibration_panel):
    # Regression test: handle_camera_event now detects on a downsampled
    # copy (see MAX_CALIBRATION_PREVIEW_DIM's own comment -- fixes a real
    # freeze when a high-res camera, e.g. an ASI678MM-class sensor, is
    # used as the main camera: detect_brightest_blob alone cost ~60ms at
    # that resolution vs ~9ms at this project's normal ~2MP main camera,
    # enough to blow the ~100ms preview budget and back up the whole Tk
    # event queue). self._latest_blob (and everything downstream: the
    # calibration sequence, auto-guide's dx_px/dy_px against frame.shape)
    # must still land in FULL-resolution pixel coordinates, not the
    # downsampled detection copy's -- otherwise every consumer of
    # _latest_blob silently breaks the moment a frame exceeds the cap.
    p = calibration_panel
    _make_camera_calibration_tab_visible(p)
    h, w = 1200, 1600  # comfortably above MAX_CALIBRATION_PREVIEW_DIM (480)
    frame = np.full((h, w), 15, dtype=np.uint8)
    true_cx, true_cy = 1000, 300
    frame[true_cy - 10:true_cy + 10, true_cx - 10:true_cx + 10] = 220
    p.handle_camera_event(CameraEvent(kind="preview_frame", payload={"pgm": frame_to_pgm(frame), "width": w, "height": h}))
    assert p._latest_blob is not None
    assert p._latest_blob.found is True
    # Within a couple pixels: the downsample stride quantizes the centroid
    # a bit, exact equality isn't the point -- landing in the right
    # coordinate SYSTEM (full-res, not the ~480px-capped detection copy) is.
    assert p._latest_blob.centroid_x == pytest.approx(true_cx, abs=5)
    assert p._latest_blob.centroid_y == pytest.approx(true_cy, abs=5)
    # The preview image itself must stay roughly bounded regardless of the
    # source frame's resolution -- this used to be a full, unbounded
    # resolution tk.PhotoImage (1600x1200 here). downsample_for_display's
    # own integer-stride rounding means the result isn't an exact ceiling
    # (534px came out of a 480px cap here), so allow some slack -- the
    # point is "roughly capped", not "full resolution" (which would be
    # 1600x1200, more than 2x over even generous slack).
    assert p._preview_image.width() <= MAX_CALIBRATION_PREVIEW_DIM * 1.5
    assert p._preview_image.height() <= MAX_CALIBRATION_PREVIEW_DIM * 1.5


def test_calibration_panel_handle_camera_event_stays_fast_on_a_finder_class_resolution(calibration_panel):
    # Regression test for a real freeze: the pre-fix code (full-resolution
    # detect_brightest_blob + a full-resolution preview PhotoImage, every
    # single preview_frame event) measured ~285ms end-to-end for a
    # finder-class frame (3840x2160, e.g. an ASI678MM used as the main
    # camera) on this machine -- enough to blow the ~100ms preview
    # interval and, once App._pump_events had a backlog of queued
    # preview_frame events to drain, freeze the whole Tk main thread. The
    # post-fix path measured ~5-10ms on the same frame. 100ms sits
    # comfortably between the two (>10x the fixed path's steady-state
    # time, <3x under the pre-fix time) -- tight enough to catch a
    # regression back to full-resolution processing, loose enough not to
    # be sensitive to normal machine-to-machine timing noise.
    import time

    p = calibration_panel
    h, w = 2160, 3840
    frame = np.full((h, w), 15, dtype=np.uint8)
    frame[1000:1020, 1600:1620] = 220
    payload = {"pgm": frame_to_pgm(frame), "width": w, "height": h}

    t0 = time.perf_counter()
    p.handle_camera_event(CameraEvent(kind="preview_frame", payload=payload))
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.1


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
    ra_payload = {
        "lag_s": 0.345, "steady_rate_arcsec_s": 1200.0, "samples": 30,
        "decel_lag_s": 0.4, "stop_command_t": 2.5, "velocity_samples": ((0.1, 100.0), (2.6, 1150.0), (3.0, 50.0)),
        "axis": "ra",
    }
    dec_payload = {**ra_payload, "steady_rate_arcsec_s": -1190.0, "axis": "dec"}
    p.handle_mount_event(WorkerEvent("mount_lag_result", {"ra": ra_payload, "dec": dec_payload}))
    assert "0.345" in p._lag_status_var.get()
    assert str(p._lag_measure_button["state"]) == "normal"
    assert p._mount_lag_var.get() == pytest.approx(0.345)


def test_calibration_panel_mount_health_updates_status_and_reenables_button(calibration_panel):
    p = calibration_panel
    p.set_connected(True)
    p._health_button.configure(state="disabled")
    p.handle_mount_event(WorkerEvent("mount_health", {
        "ra_stall_load": 0, "dec_stall_load": 3, "temperature_c": 36.694401,
        "ra_current": 28, "dec_current": 15,
    }))
    text = p._health_var.get()
    assert "36.7" in text
    assert "28" in text and "15" in text
    assert str(p._health_button["state"]) == "normal"


def test_calibration_panel_read_health_click_disables_button_and_queues_worker_command(calibration_panel):
    p = calibration_panel
    p.set_connected(True)
    p._on_read_health_click()
    assert str(p._health_button["state"]) == "disabled"
    assert "Reading" in p._health_var.get()


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
    # devices are wired up from one tab), not TransitPanel. get_optical_train
    # only feeds a REAL camera connection -- Mock mode uses this panel's own
    # focal/sensor/pixel-size fields instead (see the next test), since the
    # Exposure calc tab's optical train models a hypothetical setup for
    # exposure planning, not necessarily what the mock should simulate.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    try:
        train = OpticalTrain(aperture_mm=200, focal_length_mm=1000, barlow_multiplier=2.0, pixel_size_um=2.9)
        p = ConnectionPanel(root, mount_worker, camera_worker, lambda _connected: None, get_optical_train=lambda: train, map_widget_cls=_StubMapWidget)
        p._camera_kind_var.set("real")
        captured = {}
        p._camera_worker.connect = lambda *args, **kwargs: captured.update(kwargs, kind=args[0])
        p._on_camera_connect_click()
        assert captured["plate_scale_arcsec_per_px"] == pytest.approx(train.plate_scale_arcsec_per_px)
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_camera_connect_falls_back_to_default_plate_scale_when_fields_invalid():
    # Real mode, no optical train configured -- falls back to CameraWorker/
    # MockAsiCamera's own default.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    try:
        p = ConnectionPanel(root, mount_worker, camera_worker, lambda _connected: None, get_optical_train=lambda: None, map_widget_cls=_StubMapWidget)
        p._camera_kind_var.set("real")
        captured = {}
        p._camera_worker.connect = lambda *args, **kwargs: captured.update(kwargs, kind=args[0])
        p._on_camera_connect_click()
        assert captured["plate_scale_arcsec_per_px"] is None  # CameraWorker/MockAsiCamera fall back to their own default
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_camera_connect_in_mock_mode_uses_this_panels_own_optics_fields():
    # Mock mode: focal length / sensor size / pixel size come from
    # ConnectionPanel's own fields (defaults to the real ASI290MC + 1000mm
    # main tube specs), computed into a real plate scale -- NOT from
    # get_optical_train, even if one is configured.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    try:
        train = OpticalTrain(aperture_mm=200, focal_length_mm=2000, barlow_multiplier=1.0, pixel_size_um=5.0)
        p = ConnectionPanel(root, mount_worker, camera_worker, lambda _connected: None, get_optical_train=lambda: train, map_widget_cls=_StubMapWidget)
        assert p._camera_kind_var.get() == "mock"  # default
        captured = {}
        p._camera_worker.connect = lambda *args, **kwargs: captured.update(kwargs, kind=args[0])
        p._on_camera_connect_click()
        expected_scale = ConnectionPanel._plate_scale_arcsec_per_px(
            ConnectionPanel.MAIN_DEFAULT_FOCAL_MM, ConnectionPanel.MAIN_DEFAULT_PIXEL_UM,
        )
        assert captured["plate_scale_arcsec_per_px"] == pytest.approx(expected_scale)
        assert captured["mock_sensor_width"] == ConnectionPanel.MAIN_DEFAULT_SENSOR_W
        assert captured["mock_sensor_height"] == ConnectionPanel.MAIN_DEFAULT_SENSOR_H
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_camera_connect_in_mock_mode_recovers_cleanly_from_a_zero_focal_length():
    # Regression: a "0" focal length parses fine as a float (no ValueError
    # from the field-parsing guard), then dividing by it inside _plate_
    # scale_arcsec_per_px raised an uncaught ZeroDivisionError -- confirmed
    # to leave the connect button stuck disabled at "Connecting..." with
    # no error shown and no way to retry short of restarting the app.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    try:
        p = ConnectionPanel(root, mount_worker, camera_worker, lambda _connected: None, map_widget_cls=_StubMapWidget)
        assert p._camera_kind_var.get() == "mock"  # default
        p._main_focal_var.set("0")
        p._camera_worker.connect = lambda *a, **kw: pytest.fail("must not attempt to connect with invalid optics")

        p._on_camera_connect_click()  # must not raise

        assert str(p._camera_connect_button["state"]) == "normal"
        assert "Invalid" in p._camera_status_var.get()
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_exposure_panel_get_optical_train_returns_none_for_a_zero_focal_length():
    # Same regression, the real-camera branch's source: get_optical_train
    # already caught ValueError from float(...) parsing, and now also
    # catches the ValueError OpticalTrain.__post_init__ raises for a
    # degenerate (zero) effective focal length -- so a "0" in the field
    # yields None (the existing, already-handled "not configured" case)
    # instead of a train that would raise ZeroDivisionError on first use
    # of plate_scale_arcsec_per_px (see ConnectionPanel._on_camera_connect_
    # click, which only checks `train is not None` before using it).
    root = tk.Tk()
    root.withdraw()
    try:
        p = ExposurePanel(root)
        p._focal_var.set("0")
        assert p.get_optical_train() is None
    finally:
        root.destroy()


def test_finder_connect_in_mock_mode_recovers_cleanly_from_a_zero_focal_length():
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    finder_worker = CameraWorker()
    finder_state = FinderState()
    try:
        p = ConnectionPanel(
            root, mount_worker, camera_worker, lambda _connected: None,
            finder_worker=finder_worker, finder_state=finder_state, map_widget_cls=_StubMapWidget,
        )
        assert p._finder_kind_var.get() == "mock"  # default
        p._finder_focal_var.set("0")
        p._finder_worker.connect = lambda *a, **kw: pytest.fail("must not attempt to connect with invalid optics")

        p._on_finder_connect_click()  # must not raise

        assert str(p._finder_connect_button["state"]) == "normal"
        assert "Invalid" in p._finder_status_var.get()
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        finder_worker.shutdown()
        root.destroy()


def test_camera_and_finder_connect_push_their_real_plate_scale_into_finder_state():
    # Regression: FinderCameraPanel's "Calibrate fields" used to read a
    # separately-typed plate-scale field that could silently drift out of
    # sync with what's actually configured -- see FinderState's
    # main_plate_scale_arcsec/finder_plate_scale_arcsec docstring.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    finder_worker = CameraWorker()
    finder_state = FinderState()
    try:
        p = ConnectionPanel(
            root, mount_worker, camera_worker, lambda _connected: None,
            map_widget_cls=_StubMapWidget, finder_worker=finder_worker, finder_state=finder_state,
        )
        p._camera_worker.connect = lambda *args, **kwargs: None
        p._finder_worker.connect = lambda *args, **kwargs: None
        p._on_camera_connect_click()
        p._on_finder_connect_click()
        expected_main_scale = ConnectionPanel._plate_scale_arcsec_per_px(
            ConnectionPanel.MAIN_DEFAULT_FOCAL_MM, ConnectionPanel.MAIN_DEFAULT_PIXEL_UM,
        )
        expected_finder_scale = ConnectionPanel._plate_scale_arcsec_per_px(
            ConnectionPanel.FINDER_DEFAULT_FOCAL_MM, ConnectionPanel.FINDER_DEFAULT_PIXEL_UM,
        )
        assert finder_state.main_plate_scale_arcsec == pytest.approx(expected_main_scale)
        assert finder_state.finder_plate_scale_arcsec == pytest.approx(expected_finder_scale)
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        finder_worker.shutdown()
        root.destroy()


def test_real_finder_connect_also_pushes_the_real_plate_scale():
    # Regression: connecting a REAL finder camera used to never touch
    # FinderState.finder_plate_scale_arcsec at all (only the mock branch
    # did), silently leaving it stuck at the 1.0 dataclass default --
    # wrong by the real finder's ~1.72"/px, corrupting both the FOV
    # rectangle's size and every finder-based correction's magnitude for
    # anyone using real hardware.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    finder_worker = CameraWorker()
    finder_state = FinderState()
    try:
        p = ConnectionPanel(
            root, mount_worker, camera_worker, lambda _connected: None,
            map_widget_cls=_StubMapWidget, finder_worker=finder_worker, finder_state=finder_state,
        )
        p._finder_worker.connect = lambda *args, **kwargs: None
        p._finder_kind_var.set("real")
        p._on_finder_connect_click()
        expected_finder_scale = ConnectionPanel._plate_scale_arcsec_per_px(
            ConnectionPanel.FINDER_DEFAULT_FOCAL_MM, ConnectionPanel.FINDER_DEFAULT_PIXEL_UM,
        )
        assert finder_state.finder_plate_scale_arcsec == pytest.approx(expected_finder_scale)
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        finder_worker.shutdown()
        root.destroy()


def test_finder_focal_and_pixel_fields_stay_editable_in_real_mode():
    # The sensor W/H fields are correctly mock-only (a real camera reports
    # its own sensor size), but focal length/pixel size are never
    # auto-reported by ANY camera, real or mock -- locking them in real
    # mode (as originally implemented) left no way at all to configure
    # the real finder's plate scale.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    finder_worker = CameraWorker()
    try:
        p = ConnectionPanel(
            root, mount_worker, camera_worker, lambda _connected: None,
            map_widget_cls=_StubMapWidget, finder_worker=finder_worker,
        )
        p._finder_kind_var.set("real")
        p._update_mock_optics_state("finder")
        for widget in p._finder_optics_always_editable_widgets:
            assert str(widget["state"]) == "normal"
        for widget in p._finder_optics_widgets:
            assert str(widget["state"]) == "disabled"
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        finder_worker.shutdown()
        root.destroy()


def test_finder_calibrate_success_notifies_on_finder_calibration_ready():
    # Regression: the Transit tab's "Enable finder correction" checkbox was
    # created disabled and never re-enabled anywhere -- there was no
    # callback from the finder field calibration to TransitPanel, unlike
    # CalibrationPanel's own on_calibration_ready for auto-guiding. Field
    # calibration moved from FinderCameraPanel into CalibrationPanel (see
    # CalibrationPanel._build_finder_calibration_section) so both
    # calibrations TransitPanel's checkboxes depend on live in one tab.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    finder_state = FinderState()
    notified = []
    try:
        p = CalibrationPanel(
            root, mount_worker, camera_worker, LiveOffsets(),
            finder_state=finder_state, on_finder_calibration_ready=lambda: notified.append(True),
        )
        finder_state.last_frame = np.zeros((20, 20), dtype=np.uint8)
        finder_state.last_main_frame = np.zeros((20, 20), dtype=np.uint8)
        finder_state.calibration.calibrate_from_frames = lambda *a, **kw: setattr(
            finder_state.calibration, "calibrated", True,
        )
        p._on_calibrate_finder_fields()
        assert notified == [True]
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_finder_camera_panel_and_finder_window_share_exposure_gain_sliders():
    # Real reported bug: FinderCameraPanel (the Finder tab) and
    # FinderWindow (the floating window) each built their own independent
    # exposure/gain tk.DoubleVars -- changing one slider left the other
    # showing a stale value, with no indication the two had drifted apart
    # (same class of bug CameraControlVars was originally built to fix for
    # the main camera, see its own docstring). Passing the SAME
    # CameraControlVars instance to both -- what App now does -- must make
    # Tk's own variable-sharing keep them in lockstep automatically.
    root = tk.Tk()
    root.withdraw()
    finder_worker = CameraWorker()
    shared_vars = CameraControlVars.create(FinderCameraPanel.FINDER_DEFAULT_EXPOSURE_US, FinderCameraPanel.FINDER_DEFAULT_GAIN)
    try:
        panel = FinderCameraPanel(root, finder_worker, FinderState(), camera_vars=shared_vars)
        window = FinderWindow(root, finder_worker, FinderState(), on_sync=lambda *a: None, camera_vars=shared_vars)

        panel._camera_vars.exposure_log.set(5.0)  # 100ms
        assert window._camera_vars.exposure_log.get() == pytest.approx(5.0)
        assert "100" in window._camera_vars.exposure_value.get()

        window._camera_vars.gain.set(250)
        assert panel._camera_vars.gain.get() == pytest.approx(250)
        assert panel._camera_vars.gain_value.get() == "250"
    finally:
        finder_worker.shutdown()
        root.destroy()


def test_finder_calibrate_failure_does_not_notify_on_finder_calibration_ready():
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    finder_state = FinderState()
    notified = []
    try:
        p = CalibrationPanel(
            root, mount_worker, camera_worker, LiveOffsets(),
            finder_state=finder_state, on_finder_calibration_ready=lambda: notified.append(True),
        )
        finder_state.last_frame = np.zeros((20, 20), dtype=np.uint8)
        finder_state.last_main_frame = np.zeros((20, 20), dtype=np.uint8)

        def _raise(*_a, **_kw):
            raise ValueError("boom")

        finder_state.calibration.calibrate_from_frames = _raise
        p._on_calibrate_finder_fields()
        assert notified == []
        assert "failed" in p._finder_calib_status_var.get().lower()
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_finder_calibrate_passes_the_typed_rotation_through():
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    finder_state = FinderState()
    try:
        p = CalibrationPanel(root, mount_worker, camera_worker, LiveOffsets(), finder_state=finder_state)
        finder_state.last_frame = np.zeros((20, 20), dtype=np.uint8)
        finder_state.last_main_frame = np.zeros((20, 20), dtype=np.uint8)
        p._finder_rotation_var.set("7.5")

        captured = {}
        finder_state.calibration.calibrate_from_frames = lambda *a, **kw: captured.update(kw)
        p._on_calibrate_finder_fields()

        assert captured["rotation_deg"] == pytest.approx(7.5)
    finally:
        mount_worker.shutdown()
        camera_worker.shutdown()
        root.destroy()


def test_finder_panel_exposure_gain_commit_only_on_slider_release(monkeypatch):
    # Regression: these sliders used to fire on every drag tick
    # (command=...), unlike TransitPanel's own exposure/gain scales
    # (<ButtonRelease-1> only) -- a fast drag queued a burst of
    # set_exposure_us calls the CameraWorker then worked through one at a
    # time (see camera/worker.py's _COALESCE_LATEST_ONLY fix), reported
    # live as the finder's live feed "never" speeding back up after
    # dragging the exposure down. The worker fix alone already covers
    # this, but committing only on release is still the right UI (no
    # reason to push throwaway intermediate values to the real sensor).
    root = tk.Tk()
    root.withdraw()
    finder_worker = CameraWorker()
    try:
        p = FinderCameraPanel(root, finder_worker, FinderState())
        p._connected = True
        calls = []
        monkeypatch.setattr(finder_worker, "set_exposure_us", lambda us: calls.append(("exp", us)))
        monkeypatch.setattr(finder_worker, "set_gain", lambda g: calls.append(("gain", g)))

        # Simulate a drag: several intermediate value changes, no release.
        for log_val in (4.0, 3.5, 3.0, 2.5, 2.0):
            p._camera_vars.exposure_log.set(log_val)
        assert calls == []  # nothing committed mid-drag

        p._finder_exp_scale.event_generate("<ButtonRelease-1>")
        root.update()
        assert calls == [("exp", round(10 ** 2.0))]

        calls.clear()
        p._camera_vars.gain.set(123.0)
        p._finder_gain_scale.event_generate("<ButtonRelease-1>")
        root.update()
        assert calls == [("gain", 123)]
    finally:
        finder_worker.shutdown()
        root.destroy()


def test_finder_panel_delta_t_and_perp_nudge_controls_use_live_offsets():
    # The Finder tab got its own delta_t/perp-nudge controls (mirroring
    # TransitPanel's) so the operator doesn't have to look away from the
    # finder's wide view to nudge tracking once a pass starts -- that's
    # exactly when they're watching the finder to frame the ISS into the
    # much narrower acquisition camera.
    root = tk.Tk()
    root.withdraw()
    finder_worker = CameraWorker()
    finder_state = FinderState()
    try:
        p = FinderCameraPanel(root, finder_worker, finder_state)

        p._live_offsets.adjust_delta_t(1.0)
        dt, _ = p._live_offsets.snapshot()
        assert dt == pytest.approx(1.0)

        p._live_offsets.trigger_perp_pulse(-1.0)
        _, perp = p._live_offsets.snapshot()
        assert perp == -1.0

        # Keyboard path -- same handlers the recursive _bind_offset_keys
        # binding wires to <Up>/<Down>/<Left>/<Right>.
        p._on_finder_delta_t_key_press(0.1)
        dt, _ = p._live_offsets.snapshot()
        assert dt == pytest.approx(1.1)

        p._on_finder_perp_nudge_key_press(1.0)
        _, perp = p._live_offsets.snapshot()
        assert perp == 1.0
    finally:
        finder_worker.shutdown()
        root.destroy()


def test_transit_panel_apply_roi_pushes_current_capture_extent_into_finder_state():
    # Regression: the finder preview's FOV rectangle used to always reflect
    # the main camera's FULL sensor, even after dragging a smaller ROI --
    # see FinderState.main_sensor_width/height and main_roi_offset_row/col.
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    finder_state = FinderState()
    try:
        p = TransitPanel(root, mount_worker, camera_worker, Path("/tmp"), finder_state=finder_state)
        p._sensor_width, p._sensor_height = 1936, 1096
        p._camera_worker.set_roi = lambda *a, **kw: None

        p._apply_roi(0, 0, 1936, 1096)
        assert finder_state.main_sensor_width == 1936
        assert finder_state.main_sensor_height == 1096
        assert finder_state.main_roi_offset_row == pytest.approx(0.0)
        assert finder_state.main_roi_offset_col == pytest.approx(0.0)

        # A 400x300 ROI starting at (100, 50): centre is (300, 200), full
        # sensor centre is (968, 548) -- offset is centre - full_centre.
        p._apply_roi(100, 50, 400, 300)
        assert finder_state.main_sensor_width == 400
        assert finder_state.main_sensor_height == 300
        assert finder_state.main_roi_offset_col == pytest.approx((100 + 200) - 968)
        assert finder_state.main_roi_offset_row == pytest.approx((50 + 150) - 548)
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


def test_passes_panel_has_scheduled_and_live_sub_tabs():
    # PassesPanel was restructured into a sub-notebook (Scheduled passes /
    # Live now) -- confirms both exist and the pre-existing widgets
    # (target picker, tree, etc.) still resolve as instance attributes
    # regardless of which sub-frame they now live in.
    root = tk.Tk()
    root.withdraw()
    try:
        p = PassesPanel(root, lambda *a: None)
        tab_texts = [p._sub_notebook.tab(i, "text") for i in range(len(p._sub_notebook.tabs()))]
        assert tab_texts == ["Scheduled passes", "Live now"]
    finally:
        root.destroy()


def test_passes_panel_does_not_fetch_the_live_catalog_on_construction():
    # Regression: an early version auto-fetched the "visual" group over
    # the real network as soon as the panel was built -- every test that
    # constructs a PassesPanel (there are many, throughout this file)
    # would have silently made a real network call. Loading is manual-
    # only now (the "Reload catalog + refresh" button), same convention
    # as "Scheduled passes"' own "Refresh passes".
    root = tk.Tk()
    root.withdraw()
    try:
        with patch("am5.gui.panels.load_satellite_group_tles") as fake_load:
            p = PassesPanel(root, lambda *a: None)
            root.update()
            fake_load.assert_not_called()
        assert p._live_satellites == []
    finally:
        root.destroy()


def test_live_now_reload_populates_tree_and_status(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    try:
        p = PassesPanel(root, lambda *a: None)
        ts = load.timescale()
        satellite = EarthSatellite(_TLE_LINE1, _TLE_LINE2, "ISS (fixture)", ts)
        site = wgs84.latlon(46.18, 6.14)
        window = _window(datetime.now(timezone.utc) + timedelta(seconds=120))

        # Bypass the real background thread + network/SGP4 work -- this
        # test is about _poll_results' handling of "live_list_ready", not
        # about currently_visible_satellites/current_pass_window
        # themselves (covered directly in tests/test_ephemeris.py).
        monkeypatch.setattr(
            p, "_live_fetch_group_and_refresh",
            lambda: p._results.put(("live_list_ready", ([satellite], site, [(satellite, 45.0, 180.0, window)]))),
        )
        p._on_live_reload_click()
        for _ in range(50):
            if p._live_visible:
                break
            time.sleep(0.02)
            p._poll_results()

        assert len(p._live_visible) == 1
        row = p._live_tree.item("0")["values"]
        assert row[0] == "ISS (fixture)"
        assert row[1] == 25544
        assert row[2] == "45.0"
        assert row[3] == "180.0"
        assert "1 satellites in catalog" in p._live_status_var.get() or "1 satellite" in p._live_status_var.get()
        assert str(p._live_reload_button["state"]) == "normal"
    finally:
        root.destroy()


def test_live_now_row_click_loads_the_trajectory_like_a_scheduled_pass(monkeypatch):
    # Selecting a "Live now" row must reach the exact same on_pass_selected
    # callback a scheduled-pass row selection does -- TransitPanel etc.
    # don't need to know or care whether the pass was scheduled or live.
    root = tk.Tk()
    root.withdraw()
    selected = []
    try:
        p = PassesPanel(root, lambda *a: selected.append(a))
        ts = load.timescale()
        satellite = EarthSatellite(_TLE_LINE1, _TLE_LINE2, "ISS (fixture)", ts)
        site = wgs84.latlon(46.18, 6.14)
        window = _window(datetime.now(timezone.utc) + timedelta(seconds=120))
        p._live_site = site
        p._live_visible = [(satellite, 45.0, 180.0, window)]
        p._populate_live_tree()

        monkeypatch.setattr(
            p, "_live_compute_trajectory",
            lambda sat, win: p._results.put(("live_trajectory_ready", (
                compute_trajectory(sat, site, win.t_rise, win.t_set, step_s=1.0), win, [], sat.name,
            ))),
        )
        p._live_tree.selection_set("0")
        root.update()
        for _ in range(50):
            if selected:
                break
            time.sleep(0.02)
            p._poll_results()

        assert len(selected) == 1
        trajectory, out_window, crossings, out_site, satellite_name = selected[0]
        assert satellite_name == "ISS (fixture)"
        assert out_site is site
        assert out_window is window
    finally:
        root.destroy()


def test_live_now_sky_map_click_selects_the_same_row_as_the_list(monkeypatch):
    # The sky map is a second way to pick a satellite, not a separate
    # feature -- clicking a plotted point must drive the SAME tree
    # selection (and therefore the same on_pass_selected callback) a row
    # click would, not a parallel path that could disagree with the list.
    root = tk.Tk()
    root.withdraw()
    try:
        p = PassesPanel(root, lambda *a: None)
        ts = load.timescale()
        sat_a = EarthSatellite(_TLE_LINE1, _TLE_LINE2, "ISS (fixture)", ts)
        sat_b = EarthSatellite(_TLE_LINE1, _TLE_LINE2, "CSS (fixture)", ts)
        site = wgs84.latlon(46.18, 6.14)
        window = _window(datetime.now(timezone.utc) + timedelta(seconds=120))
        p._live_site = site
        p._live_visible = [(sat_a, 45.0, 180.0, window), (sat_b, 30.0, 90.0, window)]
        p._populate_live_tree()

        # Bypass the real background trajectory computation the resulting
        # <<TreeviewSelect>> event triggers (already covered by
        # test_live_now_row_click_loads_the_trajectory_like_a_scheduled_
        # pass) -- this test is only about which row ends up selected.
        monkeypatch.setattr(threading, "Thread", lambda target, args, daemon: _NoStartThread())

        entry_b = next(e for e, _az, _alt in p._live_sky_map._stars if e.satellite is sat_b)
        p._on_live_map_entry_selected(entry_b)
        root.update()

        assert p._live_tree.selection() == ("1",)
    finally:
        root.destroy()


def test_live_now_refresh_tick_skips_work_when_catalog_not_loaded_or_tab_hidden(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    try:
        p = PassesPanel(root, lambda *a: None)
        started = []
        monkeypatch.setattr(threading, "Thread", lambda target, daemon: started.append(target) or _NoStartThread())

        # Catalog never loaded (_live_satellites == []) -- must not spawn
        # a refresh thread even if we pretend the tab is mapped.
        monkeypatch.setattr(p._live_tab, "winfo_ismapped", lambda: True)
        p._on_live_refresh_tick()
        assert started == []

        # Catalog loaded, but tab not the visible one -- still no work.
        p._live_satellites = [EarthSatellite(_TLE_LINE1, _TLE_LINE2, "ISS (fixture)", load.timescale())]
        monkeypatch.setattr(p._live_tab, "winfo_ismapped", lambda: False)
        p._on_live_refresh_tick()
        assert started == []

        # Both conditions met -- refresh actually runs.
        monkeypatch.setattr(p._live_tab, "winfo_ismapped", lambda: True)
        p._on_live_refresh_tick()
        assert started == [p._live_refresh_visible_only]
    finally:
        root.destroy()


class _NoStartThread:
    def start(self) -> None:
        pass
