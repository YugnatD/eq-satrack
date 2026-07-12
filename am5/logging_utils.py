"""CSV telemetry logging shared by characterize.py and track_pass.py."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import IO


def open_csv(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, IO[str]]:
    fh = open(path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return writer, fh


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
