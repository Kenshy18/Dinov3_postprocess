"""Overlay rendering."""

from __future__ import annotations

from typing import Sequence
from pathlib import Path

from ..entrypoint import run_argparse_main
from ..legacy import stage_command

STAGE = "__onefile_render"


def command(args: Sequence[str]) -> list[str]:
    return stage_command(STAGE, args)


def run(args: Sequence[str], model_root: Path | None = None) -> int:
    from ..engine.render_overlay import render_main

    return run_argparse_main(render_main, args)
