import tkinter as tk
from tkinter import ttk

import pytest

from am5.gui.dialogs import confirm_phrase_dialog


def _tk_available() -> bool:
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


pytestmark = pytest.mark.skipif(not _tk_available(), reason="no Tk display available")


def _find_dialog(root: tk.Tk) -> tk.Toplevel:
    for child in root.winfo_children():
        if isinstance(child, tk.Toplevel):
            return child
    raise AssertionError("dialog did not open")


def _find_widget(widget: tk.Misc, cls: type) -> tk.Misc:
    for child in widget.winfo_children():
        if isinstance(child, cls):
            return child
        try:
            return _find_widget(child, cls)
        except AssertionError:
            continue
    raise AssertionError(f"no {cls.__name__} found under {widget}")


def _find_button(widget: tk.Misc, text: str) -> ttk.Button:
    for child in widget.winfo_children():
        if isinstance(child, ttk.Button) and child["text"] == text:
            return child
        try:
            return _find_button(child, text)
        except AssertionError:
            continue
    raise AssertionError(f"no button {text!r} found under {widget}")


def test_confirm_button_disabled_until_exact_match():
    root = tk.Tk()
    root.withdraw()
    try:
        def check_gating() -> None:
            dialog = _find_dialog(root)
            entry = _find_widget(dialog, ttk.Entry)
            confirm_button = _find_button(dialog, "Confirm")

            assert confirm_button.instate(["disabled"])
            entry.insert(0, "close enough")
            dialog.update_idletasks()
            assert confirm_button.instate(["disabled"])

            entry.delete(0, "end")
            entry.insert(0, "YES PLEASE")
            dialog.update_idletasks()
            assert not confirm_button.instate(["disabled"])

            confirm_button.invoke()

        root.after(50, check_gating)
        result = confirm_phrase_dialog(root, "title", "message", "YES PLEASE")
        assert result is True
    finally:
        root.destroy()


def test_cancel_returns_false_without_matching_text():
    root = tk.Tk()
    root.withdraw()
    try:
        def cancel() -> None:
            dialog = _find_dialog(root)
            _find_button(dialog, "Cancel").invoke()

        root.after(50, cancel)
        result = confirm_phrase_dialog(root, "title", "message", "YES PLEASE")
        assert result is False
    finally:
        root.destroy()


def test_closing_window_returns_false():
    root = tk.Tk()
    root.withdraw()
    try:
        def close() -> None:
            _find_dialog(root).destroy()

        root.after(50, close)
        result = confirm_phrase_dialog(root, "title", "message", "YES PLEASE")
        assert result is False
    finally:
        root.destroy()
