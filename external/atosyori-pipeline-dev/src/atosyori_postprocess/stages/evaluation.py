"""Exact evaluation, gap filling, and SQLite export stages."""

from __future__ import annotations

from typing import Sequence
from pathlib import Path

from ..entrypoint import run_argparse_main
from ..legacy import run_stage, stage_command

STAGE = "__onefile_evaluate"
GAP_FILL_STAGE = "__onefile_gap_fill"
UNION_TO_SQLITE_STAGE = "__onefile_union_to_sqlite"


def command(args: Sequence[str]) -> list[str]:
    return stage_command(STAGE, args)


def run(args: Sequence[str], model_root: Path | None = None) -> int:
    from ..engine.evaluate_keyframes_exact import kfeval_main

    return run_argparse_main(kfeval_main, args)


def run_gap_fill(args: Sequence[str], model_root: Path | None = None) -> int:
    from ..engine.fill_trackk_union_gaps import kffill_main

    return run_argparse_main(kffill_main, args)


def run_union_to_sqlite(args: Sequence[str], model_root: Path | None = None) -> int:
    from ..engine.union_json_to_pred_sqlite import union2sqlite_main

    return run_argparse_main(union2sqlite_main, args)
