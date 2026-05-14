#!/usr/bin/env python3
"""Compatibility wrapper for the relocated DINOv3 ROI classifier runtime."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "backend").is_dir():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return


_ensure_repo_root()

if __name__ == "__main__":
    runpy.run_module("backend.classifiers.dinov3_roi.runtime.two_stage_roi_classifier", run_name="__main__")
else:
    from backend.classifiers.dinov3_roi.runtime.two_stage_roi_classifier import *  # noqa: F401,F403,E501

