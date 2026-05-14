#!/usr/bin/env python3
"""Compatibility wrapper for tools.maintenance.inventory_cleanup_candidates."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    runpy.run_module("tools.maintenance.inventory_cleanup_candidates", run_name="__main__")
