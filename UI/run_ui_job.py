#!/usr/bin/env python3
"""Compatibility entrypoint for apps.qt_ui.run_ui_job."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.qt_ui.run_ui_job import main


if __name__ == "__main__":
    raise SystemExit(main())
