import tkinter as tk
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pytest

from am5.gui.panels import SerPlayerPanel
from camera.ser_writer import SerWriter


def _tk_available() -> bool:
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


pytestmark = pytest.mark.skipif(not _tk_available(), reason="no Tk display available")


@pytest.fixture
def root():
    r = tk.Tk()
    r.geometry("400x400")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def panel(root):
    p = SerPlayerPanel(root)
    p.pack(fill="both", expand=True)
    root.update_idletasks()
    return p


def _write_ser(path, n_frames=5, width=16, height=12, pixel_depth=8, fps=10.0):
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    dtype = np.uint8 if pixel_depth == 8 else np.uint16
    with SerWriter(path, width=width, height=height, colour_id=0, pixel_depth=pixel_depth) as writer:
        for i in range(n_frames):
            frame = np.full((height, width), i, dtype=dtype)
            writer.add_frame(frame, timestamp=t0 + timedelta(seconds=i / fps))
    return path


def test_open_file_populates_info_and_first_frame(panel, tmp_path):
    path = _write_ser(tmp_path / "a.ser", n_frames=5)
    panel._open_file(path)
    assert "16x12" in panel._info_var.get()
    assert "5 frames" in panel._info_var.get()
    assert panel._frame_label_var.get().startswith("frame 1/5")
    assert str(panel._play_button["state"]) == "normal"


def test_open_file_reports_average_fps_from_timestamps(panel, tmp_path):
    path = _write_ser(tmp_path / "b.ser", n_frames=11, fps=10.0)
    panel._open_file(path)
    assert "avg fps: 10.0" in panel._info_var.get()


def test_open_invalid_file_shows_error_and_does_not_crash(panel, tmp_path):
    bad = tmp_path / "not_ser.ser"
    bad.write_bytes(b"garbage, not a SER file" * 5)
    with patch("am5.gui.panels.messagebox.showerror") as showerror:
        panel._open_file(bad)
        showerror.assert_called_once()
    assert panel._reader is None


def test_step_advances_and_clamps_at_bounds(panel, tmp_path):
    path = _write_ser(tmp_path / "c.ser", n_frames=3)
    panel._open_file(path)
    panel._step(1)
    assert panel._frame_index == 1
    panel._step(-5)  # clamps at 0, doesn't go negative
    assert panel._frame_index == 0
    panel._step(50)  # clamps at frame_count - 1
    assert panel._frame_index == 2


def test_playback_advances_frames_and_stops_at_the_end(panel, tmp_path):
    path = _write_ser(tmp_path / "d.ser", n_frames=3)
    panel._open_file(path)
    panel._fps_var.set("1000")  # fast, so the test doesn't wait long
    panel._start_playback()
    assert panel._playing is True

    deadline = 0
    while panel._playing and deadline < 200:
        panel.update()
        panel.after(5)
        deadline += 1

    assert panel._playing is False
    assert panel._frame_index == 2  # stopped at the last frame, not wrapped


def test_play_toggle_pauses(panel, tmp_path):
    path = _write_ser(tmp_path / "e.ser", n_frames=20)
    panel._open_file(path)
    panel._fps_var.set("5")
    panel._on_play_toggle()
    assert panel._playing is True
    panel._on_play_toggle()
    assert panel._playing is False
    assert str(panel._play_button["text"]) == "Play"


def test_opening_a_second_file_closes_the_first_reader(panel, tmp_path):
    path_a = _write_ser(tmp_path / "f.ser", n_frames=2)
    path_b = _write_ser(tmp_path / "g.ser", n_frames=4, width=8, height=8)
    panel._open_file(path_a)
    first_reader = panel._reader
    panel._open_file(path_b)
    assert panel._reader is not first_reader
    assert panel._reader.frame_count == 4


def test_16bit_file_opens_and_shows_a_frame_without_crashing(panel, tmp_path):
    path = _write_ser(tmp_path / "h.ser", n_frames=3, pixel_depth=16)
    panel._open_file(path)
    assert "16-bit" in panel._info_var.get()
    assert panel._frame_label_var.get().startswith("frame 1/3")
