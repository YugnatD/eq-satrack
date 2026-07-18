import tkinter as tk

import numpy as np
import pytest

from am5.gui.finder_window import SOLVE_RETRY_ATTEMPTS, FinderWindow
from am5.gui.worker import WorkerEvent
from camera.finder import FinderState
from camera.worker import CameraEvent, CameraWorker


def _tk_available() -> bool:
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


pytestmark = pytest.mark.skipif(not _tk_available(), reason="no Tk display available")


@pytest.fixture
def window():
    root = tk.Tk()
    root.withdraw()
    finder_worker = CameraWorker()
    synced = []
    w = FinderWindow(root, finder_worker, FinderState(), on_sync=lambda ra, dec: synced.append((ra, dec)))
    w._synced = synced  # stash for tests to read
    yield w
    finder_worker.shutdown()
    root.destroy()


def _give_it_a_solve(window):
    # Bypass PlateSolver/ASTAP entirely -- these tests are about the
    # invalidation wiring around a solve result, not the solver itself.
    window._solve_btn.configure(state="disabled")
    window._sync_btn.configure(state="disabled")

    class _Result:
        success = True
        ra_deg = 123.4
        dec_deg = 56.7
        field_rotation_deg = 0.0
        pixel_scale_arcsec = 1.7

    window._on_solve_done(_Result())


def test_successful_solve_enables_sync_and_stores_the_target(window):
    _give_it_a_solve(window)
    assert window._solved_ra == pytest.approx(123.4)
    assert window._solved_dec == pytest.approx(56.7)
    assert str(window._sync_btn["state"]) == "normal"


def test_sync_click_forwards_the_solved_coordinates(window):
    _give_it_a_solve(window)
    window._on_sync_click()
    assert window._synced == [(123.4, 56.7)]


def test_exposure_gain_commit_only_on_slider_release(window, monkeypatch):
    # Regression: same fix as FinderCameraPanel's own (am5/gui/panels.py)
    # -- these sliders used to fire on every drag tick, queuing a burst of
    # set_exposure_us calls the CameraWorker then worked through one at a
    # time, reported live as the exposure "never" coming back down after
    # a fast drag.
    window._connected = True
    calls = []
    monkeypatch.setattr(window._finder_worker, "set_exposure_us", lambda us: calls.append(("exp", us)))
    monkeypatch.setattr(window._finder_worker, "set_gain", lambda g: calls.append(("gain", g)))

    for log_val in (4.0, 3.5, 3.0, 2.5, 2.0):
        window._camera_vars.exposure_log.set(log_val)
    assert calls == []

    window._exp_scale.event_generate("<ButtonRelease-1>")
    window.update()
    # _on_slider_change applies both settings together (single combined
    # commit path here, unlike FinderCameraPanel's split methods) -- still
    # exactly one commit, not one per drag tick.
    assert ("exp", round(10 ** 2.0)) in calls
    assert len(calls) == 1 or all(name in ("exp", "gain") for name, _ in calls)

    calls.clear()
    window._camera_vars.gain.set(123.0)
    window._gain_scale.event_generate("<ButtonRelease-1>")
    window.update()
    assert ("gain", 123) in calls


def test_failed_solve_clears_any_previous_solve(window):
    _give_it_a_solve(window)

    class _FailedResult:
        success = False
        message = "no match"

    window._on_solve_done(_FailedResult())
    assert window._solved_ra is None
    assert window._solved_dec is None
    assert str(window._sync_btn["state"]) == "disabled"


def test_disconnect_invalidates_a_stale_solved_target(window):
    # Regression: a solved target used to stay syncable forever after a
    # disconnect/reconnect -- see FinderWindow._invalidate_solve's own
    # docstring for the incident (Mount.sync() overwrites the mount's
    # believed position WITHOUT moving it, so a stale sync is silent
    # corruption, not just a cosmetic staleness).
    _give_it_a_solve(window)
    window.handle_camera_event(CameraEvent(kind="disconnected", payload={}))
    assert window._solved_ra is None
    assert window._solved_dec is None
    assert str(window._sync_btn["state"]) == "disabled"


@pytest.mark.parametrize("kind", ["goto_result", "jog_goto_result", "tracking_started"])
def test_a_commanded_mount_move_invalidates_a_stale_solved_target(window, kind):
    # A real GOTO/jog-to-target/tracking start means the mount is no
    # longer at the pose the solve was taken at -- the previously solved
    # coordinate must not still be sync-able afterward.
    _give_it_a_solve(window)
    window.handle_mount_event(WorkerEvent(kind=kind, payload={}))
    assert window._solved_ra is None
    assert window._solved_dec is None
    assert str(window._sync_btn["state"]) == "disabled"


def test_idle_position_polling_does_not_invalidate_a_solved_target(window):
    # position events fire continuously (2-20Hz) regardless of whether the
    # mount actually moved -- invalidating on those would disable the sync
    # button almost immediately after every solve, defeating the feature.
    _give_it_a_solve(window)
    window.handle_mount_event(WorkerEvent(kind="position", payload={"ra_hours": 1.0, "dec_deg": 2.0}))
    assert window._solved_ra == pytest.approx(123.4)
    assert str(window._sync_btn["state"]) == "normal"


def test_position_events_are_cached_as_a_plate_solve_hint(window):
    assert window._last_mount_radec is None
    window.handle_mount_event(WorkerEvent(kind="position", payload={"ra_hours": 5.0, "dec_deg": 45.0}))
    assert window._last_mount_radec == pytest.approx((75.0, 45.0))

    window.handle_mount_event(WorkerEvent(
        kind="tracking_tick", payload={"actual_ra_deg": 30.0, "actual_dec_deg": -10.0},
    ))
    assert window._last_mount_radec == pytest.approx((30.0, -10.0))

    # actual_ra_deg is only populated every error_log_every ticks (see
    # am5/tracker.py) -- an empty-string tick must not clobber the cache.
    window.handle_mount_event(WorkerEvent(
        kind="tracking_tick", payload={"actual_ra_deg": "", "actual_dec_deg": -10.0},
    ))
    assert window._last_mount_radec == pytest.approx((30.0, -10.0))


def test_solve_passes_the_mounts_last_known_position_as_a_hint(window):
    # Regression: solve_async used to be called with no hint at all,
    # forcing ASTAP into a full blind search over its whole configured
    # search_radius_deg (30 deg by default) instead of a narrow search
    # around roughly where the mount already believes it's pointing --
    # confirmed to be why solves were reported as extremely slow.
    window._latest_frame = np.zeros((10, 10), dtype=np.uint8)
    window.handle_mount_event(WorkerEvent(kind="position", payload={"ra_hours": 5.0, "dec_deg": 45.0}))

    captured = {}
    window._solver.solve_async = lambda *a, **kw: captured.update(kw)
    window._on_solve()

    assert captured["hint_ra_deg"] == pytest.approx(75.0)
    assert captured["hint_dec_deg"] == pytest.approx(45.0)


def test_solve_retries_on_a_fresh_frame_until_it_succeeds(window):
    # Regression: a single solve attempt used to be it -- if the frame
    # right after the mount stopped moving still showed real motion
    # blur/vibration, the operator had to manually retry from scratch.
    # Now retries automatically, re-reading self._latest_frame fresh on
    # each attempt (see SOLVE_RETRY_ATTEMPTS' own comment for why that
    # alone gives each retry more settling time, no extra delay needed).
    window._latest_frame = np.zeros((5, 5), dtype=np.uint8)

    class _FailedResult:
        success = False
        message = "no match"

    class _Result:
        success = True
        ra_deg = 10.0
        dec_deg = 20.0
        field_rotation_deg = 0.0
        pixel_scale_arcsec = 1.0

    call_count = 0

    def fake_solve_async(frame, tk_widget, on_done, **kw):
        nonlocal call_count
        call_count += 1
        on_done(_FailedResult() if call_count < 3 else _Result())

    window._solver.solve_async = fake_solve_async
    window._on_solve()

    assert call_count == 3
    assert window._solved_ra == pytest.approx(10.0)
    assert str(window._solve_status_var.get()).startswith("✓")


def test_solve_gives_up_after_all_retry_attempts_fail(window):
    window._latest_frame = np.zeros((5, 5), dtype=np.uint8)

    class _FailedResult:
        success = False
        message = "no match"

    call_count = 0

    def fake_solve_async(frame, tk_widget, on_done, **kw):
        nonlocal call_count
        call_count += 1
        on_done(_FailedResult())

    window._solver.solve_async = fake_solve_async
    window._on_solve()

    assert call_count == SOLVE_RETRY_ATTEMPTS
    assert window._solved_ra is None
    assert str(window._solve_status_var.get()).startswith("✗")
    assert str(window._sync_btn["state"]) == "disabled"


def test_disconnect_mid_retry_does_not_crash_the_next_attempt(window):
    # Regression, found by code audit: _attempt_solve had no None-check
    # on self._latest_frame (unlike _on_solve's own initial guard) -- a
    # camera disconnect between retry attempts sets self._latest_frame =
    # None (handle_camera_event's "disconnected" branch), and since
    # retries re-enter directly via _on_solve_attempt_done -> _attempt_
    # solve (never back through _on_solve's guard), the next attempt used
    # to crash with AttributeError: 'NoneType' object has no attribute
    # 'copy'.
    window._latest_frame = np.zeros((5, 5), dtype=np.uint8)

    class _FailedResult:
        success = False
        message = "no match"

    call_count = 0

    def fake_solve_async(frame, tk_widget, on_done, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            window._latest_frame = None  # simulate a disconnect between attempts
        on_done(_FailedResult())

    window._solver.solve_async = fake_solve_async
    window._on_solve()  # must not raise

    assert call_count == 1  # aborted on the None frame instead of attempting a 2nd retry
    assert "disconnected" in window._solve_status_var.get().lower()
