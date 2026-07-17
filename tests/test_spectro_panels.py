import time
import tkinter as tk

import pytest

from spectro.gui.panels import AcquisitionPanel, AlignmentPanel, TargetPanel, _FULL_FRAME_DISPLAY_MAX_DIM


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

    def __init__(self, dimensions: tuple[int, int] = (0, 0), site: tuple[float, float, float] = (46.18, 6.14, 400.0)):
        self._dimensions = dimensions
        self._site = site

    def get_sensor_dimensions(self):
        return self._dimensions

    def get_plate_scale_arcsec_per_px(self):
        return None  # "not set yet" -- AcquisitionPanel falls back to an arbitrary pixel range

    def get_dispersion_a_per_px(self):
        return None

    def get_site_lat_deg(self):
        return self._site[0]

    def get_site_lon_deg(self):
        return self._site[1]

    def get_site_elevation_m(self):
        return self._site[2]


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


def test_target_panel_reads_the_live_site_from_connection_panel(root):
    # Real bug: TargetPanel used to hardcode its own site lat/lon/elevation
    # instead of reading ConnectionPanel's real, user-editable fields --
    # unlike AlignmentPanel/AcquisitionPanel/ReductionPanel, which all take
    # a connection_panel reference and call its live getters (see
    # _current_site's own docstring). Changing the site in Connection tab
    # must actually reach the standard-star search and altitude chart.
    stub = _ConnectionPanelStub(site=(12.5, -34.5, 100.0))
    panel = TargetPanel(root, connection_panel=stub)
    assert panel._current_site() == (12.5, -34.5, 100.0)


def test_target_panel_falls_back_to_defaults_without_a_connection_panel(root):
    panel = TargetPanel(root, connection_panel=None)
    assert panel._current_site() == (
        TargetPanel._SITE_LAT_DEG, TargetPanel._SITE_LON_DEG, TargetPanel._SITE_ELEVATION_M,
    )


def test_target_panel_search_uses_the_live_site_not_the_hardcoded_default(root, monkeypatch):
    stub = _ConnectionPanelStub(site=(12.5, -34.5, 100.0))
    panel = TargetPanel(root, connection_panel=stub)

    seen_sites = []

    def fake_resolve_target(name):
        from spectro.catalog import Star
        return Star(name=name, ra_deg=10.0, dec_deg=20.0, vmag=5.0, spectral_type="G2V")

    def fake_find_standard_candidates(target, lat_deg, lon_deg, elevation_m):
        seen_sites.append((lat_deg, lon_deg, elevation_m))
        return []

    monkeypatch.setattr("spectro.gui.panels.resolve_target", fake_resolve_target)
    monkeypatch.setattr("spectro.gui.panels.find_standard_candidates", fake_find_standard_candidates)

    panel._search_var.set("Regulus")
    panel._on_search()
    # _search_thread runs on a background thread -- give it a moment, then
    # drain the same way _poll_results does.
    for _ in range(50):
        if seen_sites:
            break
        time.sleep(0.02)

    assert seen_sites == [(12.5, -34.5, 100.0)]
