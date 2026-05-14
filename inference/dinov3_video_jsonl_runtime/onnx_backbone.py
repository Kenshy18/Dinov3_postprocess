#!/usr/bin/env python3
"""Compatibility wrapper for the moved DINOv3 ONNX helper."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    runpy.run_module("backend.detectors.dinov3.runtime.onnx_backbone", run_name="__main__")
else:
    from backend.detectors.dinov3.runtime.onnx_backbone import *  # noqa: F401,F403
