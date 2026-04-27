"""Ellipse keyframe optimization helpers."""

from __future__ import annotations

from typing import Sequence
from pathlib import Path

from ..entrypoint import run_argparse_main
from ..legacy import stage_command

STAGE = "__onefile_optimize"


def command(args: Sequence[str]) -> list[str]:
    return stage_command(STAGE, args)


def run(args: Sequence[str], model_root: Path | None = None) -> int:
    from ..engine.optimize_keyframes_trackk_dense_recall_standalone import kftrackk_main

    return run_argparse_main(kftrackk_main, args)
