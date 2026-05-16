#!/usr/bin/env python3
"""Primary compatibility entrypoint for backend.pipeline.cli.overlay."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline.cli.overlay import *  # noqa: F401,F403
from backend.pipeline.cli.overlay import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())
