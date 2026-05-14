#!/usr/bin/env python3
"""Compatibility wrapper for the moved EVA02 folder runtime."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    runpy.run_module(
        "backend.detectors.eva02.runtime.infer_video_folder_backbone_half_amp_compile_spatialcls_sota_rich",
        run_name="__main__",
    )
else:
    from backend.detectors.eva02.runtime.infer_video_folder_backbone_half_amp_compile_spatialcls_sota_rich import *  # noqa: F401,F403,E501
