"""Compatibility imports for tools.artifacts.runtime_artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifacts.runtime_artifacts import *  # noqa: F401,F403
