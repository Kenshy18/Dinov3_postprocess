"""Polygon keyframe optimization with adaptive anchor counts."""

from __future__ import annotations

from typing import Sequence
from pathlib import Path

from ..entrypoint import run_argparse_main
from ..legacy import stage_command
from .common import with_polygon_defaults

STAGE = "__onefile_polygon_optimize"


def command(args: Sequence[str]) -> list[str]:
    return stage_command(STAGE, args)


def run(args: Sequence[str], model_root: Path | None = None) -> int:
    from ..engine.polygon_runtime import run as run_patched_polygon

    return run_patched_polygon(with_polygon_defaults(args, model_root))
