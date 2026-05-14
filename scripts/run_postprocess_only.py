#!/usr/bin/env python3
"""Compatibility entrypoint for backend.pipeline.cli.run_postprocess_only."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline.cli.run_postprocess_only import *  # noqa: F401,F403
from backend.pipeline.cli.run_postprocess_only import main


if __name__ == "__main__":
    raise SystemExit(main())
