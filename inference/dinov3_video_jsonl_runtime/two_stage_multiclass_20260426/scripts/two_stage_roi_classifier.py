#!/usr/bin/env python3
"""Compatibility wrapper for the moved DINOv3 ROI classifier helper."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    runpy.run_module(
        "backend.detectors.dinov3.runtime.two_stage_multiclass_20260426.scripts.two_stage_roi_classifier",
        run_name="__main__",
    )
else:
    from backend.detectors.dinov3.runtime.two_stage_multiclass_20260426.scripts.two_stage_roi_classifier import *  # noqa: F401,F403,E501
