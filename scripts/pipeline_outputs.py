"""Compatibility imports for backend.pipeline.pipeline_outputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline.pipeline_outputs import *  # noqa: F401,F403
