import queue
import time
from pathlib import Path

import pytest
import tkinter as tk

from am5.gui.jog_window import JogWindow
from am5.gui.panels import TransitPanel
from am5.gui.worker import MountWorker, WorkerEvent
from am5.tracker import AxisSigns
from camera.worker import CameraEvent, CameraWorker


def _tk_available() -> bool:
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


pytestmark = pytest.mark.skipif(not _tk_available(), reason="no Tk display available")


def _wait_for(worker: MountWorker, kind: str, timeout: float = 5.0) -> WorkerEvent:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = worker.events.get(timeout=0.1)
        except queue.Empty:
            continue
        if event.kind == kind:
            return event
    raise AssertionError(f"never saw a {kind!r} event within {timeout}s")


@pytest.fixture
def window():
    root = tk.Tk()
    root.withdraw()
    mount_worker = MountWorker()
    camera_worker = CameraWorker()
    axis_signs = AxisSigns(ra=1.0, dec=1.0)
    w = JogWindow(root, mount_worker, camera_worker, axis_signs)
    yield w
    mount_worker.shutdown()
    camera_worker.shutdown()
    root.destroy()


def test_position_event_updates_readout(window):
    window.handle_mount_event(WorkerEvent("position", {"ra_hours": 3.5, "dec_deg": -12.25}))
    assert "3.5000h" in window._position_var.get()
    assert "-12.2500" in window._position_var.get()


def test_tracking_tick_event_updates_readout_only_when_populated(window):
    window.handle_mount_event(WorkerEvent("tracking_tick", {"actual_ra_deg": "", "actual_dec_deg": ""}))
    assert window._position_var.get() == "RA: --  DEC: --"

    window.handle_mount_event(WorkerEvent("tracking_tick", {"actual_ra_deg": 45.0, "actual_dec_deg": 10.0}))
    assert "3.0000h" in window._position_var.get()
    assert "+10.0000" in window._position_var.get()


def test_goto_star_click_calls_jog_goto_with_the_selected_star_and_shared_axis_signs(window):
    captured = {}
    window._mount_worker.jog_goto = lambda ra_hours, dec_deg, axis_signs: captured.update(
        ra_hours=ra_hours, dec_deg=dec_deg, axis_signs=axis_signs
    )
    window._star_var.set("Vega")
    window._on_goto_star_click()

    assert captured["ra_hours"] == pytest.approx(279.234108 / 15.0)
    assert captured["dec_deg"] == pytest.approx(38.782993)
    assert captured["axis_signs"] is window._axis_signs
    assert str(window._goto_star_button["state"]) == "disabled"


def test_jog_goto_result_event_reenables_button_and_reports_status(window):
    window.set_connected(True)
    window._goto_star_button.configure(state="disabled")
    window.handle_mount_event(WorkerEvent("jog_goto_result", {"arrived": True}))
    assert str(window._goto_star_button["state"]) == "normal"
    assert window._goto_status_var.get() == "Arrived"


def test_sync_star_click_calls_sync_with_the_selected_stars_coordinates(window):
    captured = {}
    window._mount_worker.sync = lambda ra_hours, dec_deg: captured.update(ra_hours=ra_hours, dec_deg=dec_deg)
    window._star_var.set("Vega")

    window._on_sync_star_click()

    assert captured["ra_hours"] == pytest.approx(279.234108 / 15.0)
    assert captured["dec_deg"] == pytest.approx(38.782993)
    assert str(window._sync_star_button["state"]) == "disabled"
    assert "Syncing to Vega" in window._goto_status_var.get()


def test_sync_result_event_reenables_button_and_reports_status(window):
    window.set_connected(True)
    window._sync_star_button.configure(state="disabled")
    window.handle_mount_event(WorkerEvent("sync_result", {"ok": True, "message": "Synced", "ra_hours": 1.0, "dec_deg": 2.0}))
    assert str(window._sync_star_button["state"]) == "normal"
    assert window._goto_status_var.get() == "Synced"

    window.handle_mount_event(WorkerEvent("sync_result", {"ok": False, "message": "rejected", "ra_hours": 1.0, "dec_deg": 2.0}))
    assert "Sync failed" in window._goto_status_var.get()

    window.handle_mount_event(WorkerEvent("jog_goto_result", {"arrived": False}))
    assert "Did not arrive" in window._goto_status_var.get()


def test_park_click_calls_worker_park_and_sets_status(window):
    captured = []
    window._mount_worker.park = lambda: captured.append("park")
    window._on_park_click()
    assert captured == ["park"]
    assert "Parking (:hC#)" in window._park_status_var.get()


def test_park_native_click_calls_worker_park_native_and_sets_status(window):
    captured = []
    window._mount_worker.park_native = lambda: captured.append("park_native")
    window._on_park_native_click()
    assert captured == ["park_native"]
    assert "native :hP#" in window._park_status_var.get()


def test_unpark_click_calls_worker_unpark(window):
    captured = []
    window._mount_worker.unpark = lambda: captured.append("unpark")
    window._on_unpark_click()
    assert captured == ["unpark"]


def test_parked_event_enables_unpark_and_reports_status(window):
    window.set_connected(True)
    assert str(window._unpark_button["state"]) == "disabled"
    window.handle_mount_event(WorkerEvent("parked", {"method": "home", "reply": None}))
    assert window._parked is True
    assert str(window._unpark_button["state"]) == "normal"
    assert "Parked via home" in window._park_status_var.get()


def test_unparked_event_disables_unpark_and_clears_status(window):
    window.handle_mount_event(WorkerEvent("parked", {"method": "home", "reply": None}))
    window.handle_mount_event(WorkerEvent("unparked", {}))
    assert window._parked is False
    assert str(window._unpark_button["state"]) == "disabled"
    assert window._park_status_var.get() == ""


def test_motion_widgets_disabled_while_parked(window):
    # Regression: clicking a motion button (e.g. "GOTO ->") while parked
    # used to leave it disabled forever, since the worker-side handler is
    # blocked_while_parked and never emits the reply event the click
    # handler was waiting for to re-enable it. Greying these out while
    # parked prevents that click from ever happening.
    window.set_connected(True)
    window.handle_mount_event(WorkerEvent("parked", {"method": "home", "reply": None}))
    assert str(window._goto_star_button["state"]) == "disabled"
    assert str(window._jog_buttons["n"]["state"]) == "disabled"
    assert str(window._tracking_check["state"]) == "disabled"
    assert str(window._rate_entry["state"]) == "disabled"
    # Not blocked_while_parked on the worker side -- stay usable.
    assert str(window._sync_star_button["state"]) == "normal"

    window.handle_mount_event(WorkerEvent("unparked", {}))
    assert str(window._goto_star_button["state"]) == "normal"
    assert str(window._jog_buttons["n"]["state"]) == "normal"


def test_set_connected_false_resets_parked_state_and_readouts(window):
    window.handle_mount_event(WorkerEvent("position", {"ra_hours": 3.5, "dec_deg": -12.25}))
    window.handle_mount_event(WorkerEvent("parked", {"method": "home", "reply": None}))
    window.set_connected(False)
    assert window._parked is False
    assert window._position_var.get() == "RA: --  DEC: --"
    assert window._park_status_var.get() == ""
    assert str(window._unpark_button["state"]) == "disabled"


def test_tracking_checkbox_calls_worker_set_tracking(window):
    # .invoke() toggles the variable AND fires the checkbox's own command,
    # same as a real click -- unlike setting the BooleanVar directly.
    window.set_connected(True)
    captured = []
    window._mount_worker.set_tracking = lambda on: captured.append(on)
    window._tracking_check.invoke()
    assert captured == [True]
    window._tracking_check.invoke()
    assert captured == [True, False]


def test_alt_limits_toggle_calls_worker_and_shows_warning_when_disabled(window):
    captured = []
    window._mount_worker.set_altitude_limits = lambda enabled: captured.append(enabled)
    window._alt_limits_var.set(False)
    window._on_alt_limits_toggle()
    assert captured == [False]
    assert "WARNING" in window._alt_limits_warning.cget("text")

    window._alt_limits_var.set(True)
    window._on_alt_limits_toggle()
    assert captured == [False, True]
    assert window._alt_limits_warning.cget("text") == ""


def test_exposure_and_gain_controls_start_disabled(window):
    assert str(window._exposure_scale["state"]) == "disabled"
    assert str(window._gain_scale["state"]) == "disabled"


def test_camera_connected_event_enables_controls_and_configures_bounds(window):
    controls = {
        "Exposure": {"MinValue": 100, "MaxValue": 50_000, "DefaultValue": 2000},
        "Gain": {"MinValue": 0, "MaxValue": 400, "DefaultValue": 150},
    }
    window.handle_camera_event(CameraEvent(kind="connected", payload={"width": 640, "height": 480, "is_color": True, "controls": controls}))

    assert str(window._exposure_scale["state"]) == "normal"
    assert str(window._gain_scale["state"]) == "normal"
    assert window._camera_vars.gain.get() == 150
    assert "2.00 ms" == window._camera_vars.exposure_value.get()


def test_camera_disconnected_event_disables_controls(window):
    window.handle_camera_event(CameraEvent(kind="connected", payload={"width": 640, "height": 480, "is_color": True, "controls": {}}))
    window.handle_camera_event(CameraEvent(kind="disconnected", payload={}))

    assert str(window._exposure_scale["state"]) == "disabled"
    assert str(window._gain_scale["state"]) == "disabled"


def test_exposure_release_commits_to_camera_worker(window):
    captured = {}
    window._camera_worker.set_exposure_us = lambda us: captured.update(us=us)
    window._camera_vars.exposure_log.set(4.0)  # log10(10000)
    window._on_exposure_release(None)
    assert captured["us"] == 10_000


def test_gain_release_commits_to_camera_worker(window):
    captured = {}
    window._camera_worker.set_gain = lambda gain: captured.update(gain=gain)
    window._camera_vars.gain.set(250)
    window._on_gain_release(None)
    assert captured["gain"] == 250


def test_exposure_and_gain_sliders_share_state_with_a_transit_panel(window):
    # The bug report this was built for: moving one slider didn't move the
    # other -- see CameraControlVars in am5/gui/panels.py.
    transit = TransitPanel(
        window.master, window._mount_worker, window._camera_worker, Path("/tmp"), camera_vars=window._camera_vars,
    )
    window._camera_vars.gain.set(275)
    assert transit._camera_vars.gain_value.get() == "275"
    transit._camera_vars.exposure_log.set(3.5)
    assert window._camera_vars.exposure_value.get() == transit._camera_vars.exposure_value.get()


def test_arrow_key_on_window_background_jogs_the_mount(window):
    captured = []
    window._mount_worker.jog_start = lambda direction, rate_x: captured.append(("start", direction, rate_x))
    window._mount_worker.jog_stop = lambda direction: captured.append(("stop", direction))

    window.deiconify()
    window.focus_force()
    window.update()
    window.event_generate("<Right>")
    window.event_generate("<KeyRelease-Right>")
    window.update()

    assert captured == [("start", "e", 60.0), ("stop", "e")]


def test_arrow_key_press_and_release_visually_presses_the_matching_button(window):
    # A mouse click already shows Tk's built-in pressed state for free --
    # a keypress touches no widget at all otherwise, so _on_jog_key_press
    # /_on_jog_key_release manage it explicitly. See their docstrings.
    window.deiconify()
    window.focus_force()
    window.update()

    assert window._jog_buttons["e"].instate(["pressed"]) is False
    window.event_generate("<Right>")
    window.update()
    assert window._jog_buttons["e"].instate(["pressed"]) is True
    window.event_generate("<KeyRelease-Right>")
    window.update()
    assert window._jog_buttons["e"].instate(["pressed"]) is False


def test_arrow_key_still_jogs_when_the_rate_entry_has_focus(window):
    # ttk.Entry has its own Left/Right binding (move the cursor) -- without
    # _bind_jog_keys's "break", that would swallow the key and the mount
    # would never move. See the module's _bind_jog_keys docstring.
    captured = []
    window._mount_worker.jog_start = lambda direction, rate_x: captured.append(("start", direction, rate_x))

    window.deiconify()
    window._rate_entry.focus_force()
    window.update()
    assert window.focus_get() is window._rate_entry
    window._rate_entry.event_generate("<Left>")
    window.update()

    assert captured == [("start", "w", 60.0)]


def test_arrow_key_still_jogs_when_a_camera_slider_has_focus(window):
    # Same reasoning as the rate-entry case, for ttk.Scale's own
    # Up/Down/Left/Right value-nudge bindings.
    captured = []
    window._mount_worker.jog_start = lambda direction, rate_x: captured.append(("start", direction, rate_x))

    window.deiconify()
    window._exposure_scale.focus_force()
    window.update()
    assert window.focus_get() is window._exposure_scale
    window._exposure_scale.event_generate("<Up>")
    window.update()

    assert captured == [("start", "n", 60.0)]


def test_end_to_end_goto_star_against_mock_mount_converges(window):
    # Real closed-loop convergence against MockMount, same style as
    # test_jog_goto_converges_without_using_ms in test_gui_worker.py --
    # picks a star near the mock's default start (RA=3h DEC=45deg) so the
    # proportional controller converges quickly in a test.
    window._mount_worker.connect("mock", mock_seed=1)
    _wait_for(window._mount_worker, "connected")
    _wait_for(window._mount_worker, "position", timeout=3.0)

    window._mount_worker.jog_goto(ra_hours=3.2, dec_deg=46.0, axis_signs=window._axis_signs)
    result = _wait_for(window._mount_worker, "jog_goto_result", timeout=15.0)
    assert result.payload["arrived"] is True
