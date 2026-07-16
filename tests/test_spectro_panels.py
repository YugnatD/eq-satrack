import time
import tkinter as tk

import pytest

from spectro.gui.panels import AcquisitionPanel, AlignmentPanel, _FULL_FRAME_DISPLAY_MAX_DIM


def _tk_available() -> bool:
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


pytestmark = pytest.mark.skipif(not _tk_available(), reason="no Tk display available")


class _ConnectionPanelStub:
    """Minimal stand-in -- only get_sensor_dimensions()/get_plate_scale_
    arcsec_per_px() are read by AlignmentPanel/AcquisitionPanel's
    synthetic-frame path, everything else about a real ConnectionPanel is
    irrelevant here."""

    def __init__(self, dimensions: tuple[int, int]):
        self._dimensions = dimensions

    def get_sensor_dimensions(self):
        return self._dimensions

    def get_plate_scale_arcsec_per_px(self):
        return None  # "not set yet" -- AcquisitionPanel falls back to an arbitrary pixel range

    def get_dispersion_a_per_px(self):
        return None


class _Event:
    def __init__(self, xdata: float, ydata: float):
        self.xdata = xdata
        self.ydata = ydata


@pytest.fixture
def root():
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


def test_alignment_panel_displays_a_downsampled_frame_but_keeps_full_res_extent(root):
    # Regression: AlignmentPanel._redraw_current used to hand the FULL
    # sensor-resolution frame straight to matplotlib's imshow on every
    # tick -- fine at this project's real ~1936x1096 main camera, but
    # measured at ~450-480ms/redraw at a finder-class 3840x2160 (vs
    # ~35ms after downsampling first), the same class of freeze already
    # fixed for the ISS tracker's own live previews (see camera/finder.py's
    # downsample_for_display). Applied here preventively -- spectro's
    # FRAME ACQUISITION is still synthetic, but there's no reason to wait
    # for a wider camera to hit the same wall.
    panel = AlignmentPanel(root, connection_panel=_ConnectionPanelStub((3840, 2160)), live_camera_feed=None)
    panel._on_new_demo_frame()

    assert panel._last_frame.shape == (2160, 3840)  # stored state stays full-resolution

    panel._redraw_current()
    images = panel._ax.get_images()
    assert len(images) == 1
    displayed = images[0].get_array()
    # downsample_for_display's integer-stride rounding isn't an exact
    # ceiling (960px came out of a 900px cap here) -- allow some slack,
    # the point is "roughly capped", not "full resolution" (3840x2160,
    # more than 2x over even generous slack).
    assert displayed.shape[0] <= _FULL_FRAME_DISPLAY_MAX_DIM * 1.5
    assert displayed.shape[1] <= _FULL_FRAME_DISPLAY_MAX_DIM * 1.5
    # extent must still span the FULL sensor, not the downsampled array's
    # own shape -- this is what keeps click-to-coordinate marking correct.
    assert list(images[0].get_extent()) == [0, 3840, 2160, 0]


def test_alignment_panel_order0_click_lands_in_full_resolution_coordinates(root):
    panel = AlignmentPanel(root, connection_panel=_ConnectionPanelStub((3840, 2160)), live_camera_feed=None)
    panel._on_new_demo_frame()

    panel._mode = "order0"
    panel._on_canvas_press(_Event(3000.0, 1800.0))

    # A click well outside a downsampled array's own pixel bounds (e.g. a
    # ~900px display copy) must still land at the true full-resolution
    # position -- proves extent, not the displayed array's shape, governs
    # the click coordinate space.
    assert panel._order0_xy == (3000.0, 1800.0)


def test_alignment_panel_redraw_is_fast_at_a_finder_class_resolution(root):
    panel = AlignmentPanel(root, connection_panel=_ConnectionPanelStub((3840, 2160)), live_camera_feed=None)
    panel._on_new_demo_frame()

    t0 = time.perf_counter()
    panel._redraw_current()
    elapsed = time.perf_counter() - t0

    # Comfortably between the fixed path's measured ~35ms and the old
    # full-resolution path's measured ~460ms on this machine.
    assert elapsed < 0.2


def test_acquisition_panel_full_frame_displays_a_downsampled_frame_but_keeps_full_res_extent(root):
    panel = AcquisitionPanel(
        root, role="target", seed=1, get_star=lambda: None,
        connection_panel=_ConnectionPanelStub((3840, 2160)),
    )
    panel._refresh_full_frame()

    assert panel._last_full_frame.shape == (2160, 3840)

    images = panel._full_frame_ax.get_images()
    assert len(images) == 1
    displayed = images[0].get_array()
    # downsample_for_display's integer-stride rounding isn't an exact
    # ceiling (960px came out of a 900px cap here) -- allow some slack,
    # the point is "roughly capped", not "full resolution" (3840x2160,
    # more than 2x over even generous slack).
    assert displayed.shape[0] <= _FULL_FRAME_DISPLAY_MAX_DIM * 1.5
    assert displayed.shape[1] <= _FULL_FRAME_DISPLAY_MAX_DIM * 1.5
    assert list(images[0].get_extent()) == [0, 3840, 2160, 0]


def test_downsample_is_a_no_op_below_the_display_cap(root):
    # Small frames (this project's own mock default, 640x480, or a real
    # ~1936x1096 main camera) must render at their native resolution --
    # the cap only ever shrinks, never upscales or otherwise distorts.
    panel = AlignmentPanel(root, connection_panel=_ConnectionPanelStub((640, 480)), live_camera_feed=None)
    panel._on_new_demo_frame()

    panel._redraw_current()
    displayed = panel._ax.get_images()[0].get_array()
    assert displayed.shape == (480, 640)
