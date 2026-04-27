"""K1 exact + K2 V5 routed ellipse inference."""

from __future__ import annotations

from typing import Sequence
from pathlib import Path

from ..entrypoint import run_argparse_main
from ..legacy import stage_command
from .common import with_ellipse_defaults

STAGE = "__onefile_infer"


def command(args: Sequence[str]) -> list[str]:
    return stage_command(STAGE, args)


def run(args: Sequence[str], model_root: Path | None = None) -> int:
    from ..engine.ellipse_inference import infer_main

    return run_argparse_main(infer_main, with_ellipse_defaults(args, model_root))
