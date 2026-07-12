#!/usr/bin/env python3
"""tkinter GUI for ISS tracking: browse upcoming passes, jog/GOTO the mount
manually, and run a live-adjustable tracking pass with a real-time error
plot. A front end over the same am5/ package used by characterize.py and
track_pass.py — connection (mock/serial/TCP) is chosen inside the app, on
the Connection tab.

Usage:
    python3 gui.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from am5.gui.app import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="logs", help="directory for tracking-pass CSV telemetry")
    args = parser.parse_args()
    run(Path(args.out_dir))


if __name__ == "__main__":
    main()
