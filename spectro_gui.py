#!/usr/bin/env python3
"""Frontend mockup for a Star Analyser spectroscopy app -- reuses am5/mount.py,
camera/, and am5/gui/theme.py from the satellite-tracking project, but nothing
in spectro/ is wired to real hardware yet. A showcase of the intended layout
(Connection, Target & standard star, Capture, Spectrum) to review before any
of it is connected.

Usage:
    python3 spectro_gui.py
"""

from __future__ import annotations

from spectro.gui.app import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
