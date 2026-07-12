from am5.live_input import KeyboardInput
from am5.tracker import LiveOffsets


def test_handle_key_delta_t_nudges():
    offsets = LiveOffsets()
    ki = KeyboardInput(offsets)

    ki._handle_key("]")
    assert offsets.delta_t_s == 0.1
    ki._handle_key("[")
    assert offsets.delta_t_s == 0.0
    ki._handle_key("}")
    assert offsets.delta_t_s == 1.0
    ki._handle_key("{")
    assert offsets.delta_t_s == 0.0


def test_handle_key_perp_pulse():
    offsets = LiveOffsets()
    ki = KeyboardInput(offsets)

    ki._handle_key("d")
    _, perp = offsets.snapshot()
    assert perp == 1.0

    ki._handle_key("a")
    _, perp = offsets.snapshot()
    assert perp == -1.0


def test_handle_key_quit_callback():
    calls = []
    offsets = LiveOffsets()
    ki = KeyboardInput(offsets, on_quit=lambda: calls.append(True))

    ki._handle_key("q")
    assert calls == [True]


def test_handle_key_unknown_is_noop():
    offsets = LiveOffsets()
    ki = KeyboardInput(offsets)
    ki._handle_key("z")
    assert offsets.delta_t_s == 0.0
    _, perp = offsets.snapshot()
    assert perp == 0.0
