"""Dark theme applied once at startup (see App.__init__) -- a single
source of truth for colors so raw tk widgets (Text, Canvas, Toplevel --
ttk doesn't style these) and matplotlib figures (which ignore ttk styling
entirely) match the ttk theme instead of showing up as bright white boxes
in an otherwise dark window.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk


@dataclass(frozen=True)
class Palette:
    bg: str = "#1e1e1e"  # window / frame background
    bg_alt: str = "#252526"  # LabelFrame interior, unselected tab
    bg_widget: str = "#2d2d30"  # entry / scale / combobox field
    bg_selected: str = "#0e4a78"  # selected tab, active button
    fg: str = "#d4d4d4"  # primary text
    fg_dim: str = "#8a8a8a"  # secondary / hint text (was "#666" on a light bg)
    accent: str = "#4fa3ff"  # links, focus ring, progress
    accent_ok: str = "#5fd07a"  # connected / success (was "#0a0")
    accent_warn: str = "#e5a333"  # countdowns / warnings (was "#a60")
    accent_bad: str = "#e06c75"  # errors / disconnected (was "red")
    border: str = "#3c3c3c"


PALETTE = Palette()


def apply_dark_theme(root: tk.Tk) -> Palette:
    p = PALETTE
    style = ttk.Style(root)
    # "clam" is the most stylable built-in ttk theme (unlike the native
    # platform themes, which ignore most color overrides) -- the base every
    # other override below builds on.
    style.theme_use("clam")

    root.configure(bg=p.bg)

    style.configure(
        ".", background=p.bg, foreground=p.fg, fieldbackground=p.bg_widget,
        bordercolor=p.border, lightcolor=p.bg, darkcolor=p.bg, troughcolor=p.bg_alt,
        focuscolor=p.accent, insertcolor=p.fg,
    )
    style.configure("TFrame", background=p.bg)
    style.configure("TLabelframe", background=p.bg, bordercolor=p.border, relief="groove")
    style.configure("TLabelframe.Label", background=p.bg, foreground=p.fg_dim)
    style.configure("TLabel", background=p.bg, foreground=p.fg)
    style.configure("TPanedwindow", background=p.bg)

    style.configure("TButton", background=p.bg_widget, foreground=p.fg, bordercolor=p.border, padding=(8, 4))
    style.map(
        "TButton",
        background=[("disabled", p.bg), ("pressed", p.bg_selected), ("active", p.bg_selected)],
        foreground=[("disabled", p.fg_dim)],
    )
    # JogWindow's N/S/E/W/stop buttons -- bigger glyphs read better than
    # the default button font at that size.
    style.configure("Jog.TButton", font=("", 14))

    style.configure("TEntry", fieldbackground=p.bg_widget, foreground=p.fg, bordercolor=p.border, insertcolor=p.fg)
    style.map("TEntry", fieldbackground=[("disabled", p.bg_alt)], foreground=[("disabled", p.fg_dim)])

    style.configure("TCombobox", fieldbackground=p.bg_widget, foreground=p.fg, background=p.bg_widget, arrowcolor=p.fg_dim)
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", p.bg_widget), ("disabled", p.bg_alt)],
        foreground=[("readonly", p.fg), ("disabled", p.fg_dim)],
    )
    root.option_add("*TCombobox*Listbox.background", p.bg_widget)
    root.option_add("*TCombobox*Listbox.foreground", p.fg)
    root.option_add("*TCombobox*Listbox.selectBackground", p.bg_selected)

    style.configure("TCheckbutton", background=p.bg, foreground=p.fg)
    style.map("TCheckbutton", background=[("active", p.bg)], foreground=[("disabled", p.fg_dim)])
    style.configure("TRadiobutton", background=p.bg, foreground=p.fg)
    style.map("TRadiobutton", background=[("active", p.bg)], foreground=[("disabled", p.fg_dim)])

    style.configure("TScale", background=p.bg, troughcolor=p.bg_widget)
    style.map("TScale", background=[("disabled", p.bg)])

    style.configure("TSeparator", background=p.border)

    style.configure("TNotebook", background=p.bg, bordercolor=p.border, tabmargins=(2, 4, 2, 0))
    style.configure("TNotebook.Tab", background=p.bg_alt, foreground=p.fg_dim, padding=(14, 7))
    style.map(
        "TNotebook.Tab",
        background=[("selected", p.bg_selected)],
        foreground=[("selected", p.fg)],
        expand=[("selected", (1, 1, 1, 0))],
    )

    style.configure("Treeview", background=p.bg_widget, fieldbackground=p.bg_widget, foreground=p.fg, bordercolor=p.border)
    style.map("Treeview", background=[("selected", p.bg_selected)], foreground=[("selected", p.fg)])
    style.configure("Treeview.Heading", background=p.bg_alt, foreground=p.fg_dim, bordercolor=p.border)

    style.configure("TScrollbar", background=p.bg_alt, troughcolor=p.bg, bordercolor=p.border, arrowcolor=p.fg_dim)

    return p


def style_axes(fig, ax, palette: Palette = PALETTE) -> None:
    """Applies the dark palette to a matplotlib Figure/Axes -- matplotlib
    ignores ttk theming entirely, so left alone a plot renders as a bright
    white rectangle in the middle of an otherwise dark window."""
    fig.patch.set_facecolor(palette.bg)
    ax.set_facecolor(palette.bg_alt)
    ax.tick_params(colors=palette.fg_dim, labelcolor=palette.fg_dim)
    ax.xaxis.label.set_color(palette.fg_dim)
    ax.yaxis.label.set_color(palette.fg_dim)
    ax.title.set_color(palette.fg)
    for spine in ax.spines.values():
        spine.set_color(palette.border)
    ax.grid(True, color=palette.border, linewidth=0.6, alpha=0.6)
