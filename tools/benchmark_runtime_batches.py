#!/usr/bin/env python3
"""Compatibility entrypoint for tools.setup.benchmark_runtime_batches."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.setup.benchmark_runtime_batches import main


if __name__ == "__main__":
    raise SystemExit(main())
