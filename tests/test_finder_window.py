import tkinter as tk

import pytest

from am5.gui.finder_window import FinderWindow
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
