#!/usr/bin/env python
"""Compatibility entrypoint for the DINOv3 cascade training module."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    runpy.run_module("training.dinov3.train_dinov3_cascade_unified", run_name="__main__")
else:
    from training.dinov3.train_dinov3_cascade_unified import *  # noqa: F401,F403
