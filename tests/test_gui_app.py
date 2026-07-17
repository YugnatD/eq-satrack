import queue
import tkinter as tk

import pytest

from am5.gui.app import App, _drain_coalescing_preview_frames
from am5.gui.worker import WorkerEvent
from camera.worker import CameraEvent


def _tk_available() -> bool:
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


pytestmark = pytest.mark.skipif(not _tk_available(), reason="no Tk display available")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    # Module-scoped: a full App() builds every panel (dozens of
    # tk.Variables) plus two MountWorker/CameraWorker background threads --
    # five of those per test run (one per test in this file) measurably
    # adds to the Tk/thread churn the whole suite accumulates across ~170
    # tests, which is what the flaky tests documented in
    # tests/test_gui_worker.py (thread/resource contention under the full
    # suite) are sensitive to. The tests below only ever read shared-state
    # identity or apply additive, non-conflicting mutations (see each
    # test's own reasoning), so a single shared instance is safe.
    root = tk.Tk()
    root.withdraw()
    a = App(root, tmp_path_factory.mktemp("app"))
    yield a
    a.worker.shutdown()
    a.camera_worker.shutdown()
    a.finder_worker.shutdown()
    root.destroy()


def test_jog_window_shares_the_same_axis_signs_instance_as_transit_panel(app):
    # Same object, not a copy -- a (re)calibration mutating one must be
    # visible from the other without any extra sync code, see
    # TransitPanel.set_axis_signs and JogWindow.__init__.
    assert app.jog_window._axis_signs is app.axis_signs
    assert app.transit_panel._axis_signs is app.axis_signs
    assert app.calibration_panel._axis_signs is app.axis_signs


def test_finder_panel_shares_the_same_live_offsets_instance_as_transit_panel(app):
    # A delta_t/perp nudge made from the Finder tab (added so the operator
    # doesn't have to look away from the finder's wide view right when
    # framing the ISS into the main camera matters most) must land in the
    # SAME LiveOffsets the active tracking loop and TransitPanel's own
    # controls read -- see FinderCameraPanel.__init__'s own comment.
    assert app.finder_panel._live_offsets is app.live_offsets
    assert app.transit_panel._offsets is app.live_offsets


def test_jog_window_starts_hidden_and_show_button_reveals_it(app):
    assert app.jog_window.state() == "withdrawn"
    app._show_jog_window()
    assert app.jog_window.state() != "withdrawn"


def test_calibration_done_event_updates_axis_signs_seen_by_jog_window(app):
    app._pump_events()  # drain startup events first
    app.worker.events.put(WorkerEvent("calibration_done", {"ra_sign": -1.0, "dec_sign": 1.0}))
    app._pump_events()

    assert app.jog_window._axis_signs.ra == -1.0
    assert app.jog_window._axis_signs.dec == 1.0


def test_calibration_done_event_carries_pier_side_into_axis_signs(app):
    app._pump_events()  # drain startup events first
    try:
        app.worker.events.put(WorkerEvent("calibration_done", {"ra_sign": 1.0, "dec_sign": 1.0, "pier_side": "E"}))
        app._pump_events()
        assert app.axis_signs.calibrated_pier_side == "E"
    finally:
        app.worker.events.put(WorkerEvent("calibration_done", {"ra_sign": -1.0, "dec_sign": 1.0, "pier_side": None}))
        app._pump_events()  # reset -- app fixture is module-scoped, shared with other tests


def test_connection_change_reaches_jog_window_and_calibration_panel(app):
    # Regression coverage for the parked-button-gating fix: both need to
    # know "connected" to grey their motion buttons out correctly, not
    # just TransitPanel.
    try:
        app._on_connection_change(True)
        assert app.jog_window._connected is True
        assert app.calibration_panel._connected is True
    finally:
        app._on_connection_change(False)  # reset -- app fixture is module-scoped, shared with other tests


def test_calibration_panel_shares_the_same_auto_guide_var_as_transit_panel(app):
    # The checkbox lives in the Transit tab; CalibrationPanel reads the same
    # var to decide whether to apply a detected correction.
    assert app.calibration_panel._auto_guide_var is app.auto_guide_var
    assert app.transit_panel._auto_guide_var is app.auto_guide_var


def test_calibration_panel_calibration_ready_enables_transit_panels_checkbox(app):
    assert str(app.transit_panel._auto_guide_check["state"]) == "disabled"
    app.calibration_panel._on_calibration_ready()
    assert str(app.transit_panel._auto_guide_check["state"]) == "normal"


def test_calibration_panel_shares_the_same_mount_lag_var_as_transit_panel(app):
    # CalibrationPanel's "Measure mount lag" writes it; TransitPanel's
    # start/simulate read it into TrackingConfig.mount_lag_s -- same object
    # so a measurement is immediately picked up without extra sync code.
    assert app.calibration_panel._mount_lag_var is app.mount_lag_var
    assert app.transit_panel._mount_lag_var is app.mount_lag_var


def test_mount_lag_result_event_updates_the_shared_mount_lag_var(app):
    app._pump_events()  # drain startup events first
    ra_payload = {
        "lag_s": 0.42, "steady_rate_arcsec_s": 900.0, "samples": 20,
        "decel_lag_s": 0.5, "stop_command_t": 2.0, "velocity_samples": ((0.1, 90.0), (2.1, 850.0)),
        "axis": "ra",
    }
    dec_payload = {**ra_payload, "axis": "dec"}
    app.worker.events.put(WorkerEvent("mount_lag_result", {"ra": ra_payload, "dec": dec_payload}))
    app._pump_events()

    assert app.mount_lag_var.get() == pytest.approx(0.42)
    assert app.transit_panel._mount_lag_var.get() == pytest.approx(0.42)


def test_camera_connected_event_reaches_jog_windows_exposure_gain_controls(app):
    assert str(app.jog_window._exposure_scale["state"]) == "disabled"
    app.camera_worker.events.put(CameraEvent(kind="connected", payload={"width": 640, "height": 480, "is_color": True, "controls": {}}))
    app._pump_events()

    assert str(app.jog_window._exposure_scale["state"]) == "normal"
    assert str(app.jog_window._gain_scale["state"]) == "normal"


def test_finder_disconnected_event_clears_a_stale_blob_detection(app):
    # Regression: a dropped finder connection used to leave the last
    # detected ISS blob in place forever, letting _maybe_apply_finder_
    # correction keep nudging the mount from stale data -- see
    # FinderState.reset_blob.
    app.finder_state.blob_found = True
    app.finder_state.last_blob_row = 42.0
    app.finder_state.last_blob_col = 17.0

    app.finder_worker.events.put(CameraEvent(kind="disconnected", payload={}))
    app._pump_events()

    assert app.finder_state.blob_found is False
    assert app.finder_state.last_blob_row is None
    assert app.finder_state.last_blob_col is None


def test_drain_coalescing_preview_frames_keeps_only_the_last_of_a_backlog():
    # Regression: if _pump_events ever falls behind (a slow one-off redraw
    # elsewhere on the Tk thread, e.g. TransitPanel's own sky-map redraw
    # when Simulate is clicked), the camera worker keeps producing
    # preview_frame events in the background regardless -- rendering each
    # stale one in turn (a real tk.PhotoImage + canvas update per panel,
    # confirmed via profiling as the single largest cost in a normal
    # _pump_events call) only makes an existing backlog worse. Only the
    # LAST preview_frame in a backlog was ever going to be visible anyway.
    q: "queue.Queue" = queue.Queue()
    q.put(CameraEvent(kind="preview_frame", payload={"n": 1}))
    q.put(CameraEvent(kind="preview_frame", payload={"n": 2}))
    q.put(CameraEvent(kind="stats", payload={"fps": 10.0}))
    q.put(CameraEvent(kind="preview_frame", payload={"n": 3}))
    q.put(CameraEvent(kind="preview_frame", payload={"n": 4}))

    events = _drain_coalescing_preview_frames(q)

    kinds_and_payloads = [(e.kind, e.payload) for e in events]
    assert kinds_and_payloads == [
        ("preview_frame", {"n": 2}),
        ("stats", {"fps": 10.0}),
        ("preview_frame", {"n": 4}),
    ]
    assert q.empty()


def test_drain_coalescing_preview_frames_passes_through_non_preview_events_untouched():
    q: "queue.Queue" = queue.Queue()
    q.put(CameraEvent(kind="connected", payload={"width": 640, "height": 480}))
    q.put(CameraEvent(kind="log", payload={"message": "hello"}))

    events = _drain_coalescing_preview_frames(q)

    assert [e.kind for e in events] == ["connected", "log"]


def test_pump_events_only_renders_the_last_of_several_queued_camera_preview_frames(app):
    # Integration-level version of the two tests above: pushing a backlog
    # of preview_frame events straight onto the real CameraWorker queue
    # and pumping once must leave the panel showing the LAST frame, not
    # stall trying to render every one of them.
    from camera.worker import frame_to_pgm
    import numpy as np

    # TransitPanel.handle_camera_event now skips building the preview
    # PhotoImage entirely while its canvas isn't actually mapped (real CPU
    # cost for a frame nobody can see -- a ttk.Notebook only maps the
    # selected tab's widgets), so this needs the Transit tab actually
    # selected and the root actually shown, not the usual withdrawn root
    # this fixture defaults to.
    app.root.deiconify()
    app.notebook.select(app.transit_panel)
    app.root.update()

    # Distinguished by PIXEL VALUE, not size: a real (now-mapped) canvas
    # has its own actual pixel dimensions, so fit_pgm_to_canvas scales
    # both frames to fit it -- comparing _display_w/_display_h against the
    # raw source size no longer holds once real scaling is involved (that
    # assertion only ever passed because the canvas was never laid out
    # under the old always-withdrawn test setup, an incidental zoom=1
    # fallback, not a deliberate guarantee). Checking which frame's actual
    # content made it to screen is what this test is really about.
    stale = frame_to_pgm(np.full((10, 10), 50, dtype=np.uint8))
    fresh = frame_to_pgm(np.full((20, 20), 200, dtype=np.uint8))
    app.camera_worker.events.put(CameraEvent(kind="preview_frame", payload={"pgm": stale, "width": 10, "height": 10}))
    app.camera_worker.events.put(CameraEvent(kind="preview_frame", payload={"pgm": fresh, "width": 20, "height": 20}))

    app._pump_events()

    assert app.transit_panel._preview_image.get(0, 0) == (200, 200, 200)
