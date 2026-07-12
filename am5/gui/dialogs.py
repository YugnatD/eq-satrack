"""Modal confirmation dialogs requiring exact text entry — the GUI
equivalent of am5/safety.py's confirm_tube_removed(), which blocks on
input() and can't be used from inside a Tk mainloop. Same
friction-on-purpose philosophy (typed phrase, not just an OK click), same
warnings, adapted for a dialog instead of a terminal prompt.

(The GUI's ready-to-track confirmation before ARM was removed at the
operator's request — arming/starting early and letting the loop wait for
the pass is now the supported workflow, see TransitPanel._check_pass_timing
and Trajectory.interpolate's zero-rate-outside-window clamp.)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

TUBE_REMOVED_TITLE = "Safety check — tube removed?"
TUBE_REMOVED_MESSAGE = (
    "This will drive the mount at up to 6 deg/s (1440x sidereal).\n"
    "The OTA (tube) MUST be removed from the mount before proceeding.\n"
    "An unbalanced or loaded mount slewing at speed is a real hazard\n"
    "to the equipment and to anyone nearby."
)
TUBE_REMOVED_PHRASE = "TUBE REMOVED"

def confirm_phrase_dialog(parent: tk.Misc, title: str, message: str, required_phrase: str) -> bool:
    """Blocking modal: True only if the user typed `required_phrase` exactly
    and clicked Confirm. Cancel or closing the window returns False."""
    result = {"confirmed": False}
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    dialog.resizable(False, False)

    ttk.Label(dialog, text=message, justify="left").pack(padx=16, pady=(16, 8))
    ttk.Label(dialog, text=f"Type exactly: {required_phrase}").pack(padx=16, pady=(0, 4))

    entry_var = tk.StringVar()
    entry = ttk.Entry(dialog, textvariable=entry_var, width=len(required_phrase) + 4)
    entry.pack(padx=16, pady=(0, 8))
    entry.focus_set()

    button_frame = ttk.Frame(dialog)
    button_frame.pack(padx=16, pady=(0, 16), fill="x")

    def on_confirm() -> None:
        result["confirmed"] = True
        dialog.destroy()

    def on_cancel() -> None:
        dialog.destroy()

    confirm_button = ttk.Button(button_frame, text="Confirm", command=on_confirm, state="disabled")
    confirm_button.pack(side="right", padx=(4, 0))
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="right")

    def on_entry_change(*_args: object) -> None:
        confirm_button.state(["!disabled"] if entry_var.get() == required_phrase else ["disabled"])

    entry_var.trace_add("write", on_entry_change)
    dialog.protocol("WM_DELETE_WINDOW", on_cancel)
    dialog.bind("<Return>", lambda _e: on_confirm() if entry_var.get() == required_phrase else None)
    dialog.bind("<Escape>", lambda _e: on_cancel())

    parent.wait_window(dialog)
    return result["confirmed"]


def confirm_tube_removed_dialog(parent: tk.Misc) -> bool:
    return confirm_phrase_dialog(parent, TUBE_REMOVED_TITLE, TUBE_REMOVED_MESSAGE, TUBE_REMOVED_PHRASE)
