"""Importable engine modules extracted from the legacy standalone file.

The extracted modules keep the original function names and hidden-stage logic,
while making each subsystem importable for development.
"""

from __future__ import annotations

import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

