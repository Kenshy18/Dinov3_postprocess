"""Low-level bridge to the validated legacy engine."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .settings import LEGACY_ENGINE


def legacy_command(args: Sequence[str], *, python: str | None = None) -> list[str]:
    return [python or sys.executable, str(LEGACY_ENGINE), *[str(arg) for arg in args]]


def stage_command(stage_name: str, args: Sequence[str], *, python: str | None = None) -> list[str]:
    return legacy_command([stage_name, *args], python=python)


def run_legacy(args: Sequence[str], *, python: str | None = None) -> int:
    return subprocess.run(legacy_command(args, python=python), check=False).returncode


def run_stage(stage_name: str, args: Sequence[str], *, python: str | None = None) -> int:
    return subprocess.run(stage_command(stage_name, args, python=python), check=False).returncode


def engine_exists() -> bool:
    return Path(LEGACY_ENGINE).is_file()
