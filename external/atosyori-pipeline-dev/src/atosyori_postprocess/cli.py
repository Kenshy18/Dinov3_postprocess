"""Command line interface for development and operations."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from . import doctor, pipeline
from .engine.registry import import_all
from .models import copy_from_legacy_root
from .smoke import run_polygon_smoke
from .stages import STAGES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atosyori-postprocess")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full post-processing pipeline.")
    input_group = run_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-sqlite", type=Path)
    input_group.add_argument("--input-jsonl", type=Path)
    run_parser.add_argument("--input-video", type=Path)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--intervals", default="3")
    run_parser.add_argument("--default-shape-mode", choices=("ellipse", "polygon"))
    run_parser.add_argument("--class-policy-json", type=Path)
    run_parser.add_argument("--model-root", type=Path)
    run_parser.add_argument("--k2-run-dir", type=Path)
    run_parser.add_argument("--polygon-point-predictor-model-dir", type=Path)
    run_parser.add_argument("--k2-device", default="auto")
    run_parser.add_argument("--polygon-predictor-device", default="auto")
    run_parser.add_argument("--render-overlays", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("engine_args", nargs=argparse.REMAINDER)

    stage_parser = subparsers.add_parser("stage", help="Run one feature stage directly.")
    stage_parser.add_argument("--model-root", type=Path)
    stage_parser.add_argument("name", choices=sorted(STAGES))
    stage_parser.add_argument("stage_args", nargs=argparse.REMAINDER)

    copy_parser = subparsers.add_parser("copy-models", help="Copy checkpoints from the old standalone directory.")
    copy_parser.add_argument("--source-root", type=Path, required=True)
    copy_parser.add_argument("--model-root", type=Path)
    copy_parser.add_argument("--replace", action="store_true")

    doctor_parser = subparsers.add_parser("doctor", help="Check runtime dependencies and model paths.")
    doctor_parser.add_argument("--model-root", type=Path)

    subparsers.add_parser("engine-check", help="Import all extracted engine modules.")

    sample_parser = subparsers.add_parser("make-sample", help="Create a tiny synthetic tracked SQLite.")
    sample_parser.add_argument("--output", type=Path, default=Path("output/dev_sample.sqlite"))
    sample_parser.add_argument("--frames", type=int, default=8)

    smoke_parser = subparsers.add_parser("smoke", help="Run a small local smoke check.")
    smoke_parser.add_argument("--work-dir", type=Path, default=Path("output/smoke"))
    smoke_parser.add_argument("--force", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return pipeline.run(args)
    if args.command == "stage":
        stage_args = list(args.stage_args)
        if stage_args and stage_args[0] == "--":
            stage_args = stage_args[1:]
        return STAGES[args.name].run(stage_args, model_root=args.model_root)
    if args.command == "copy-models":
        copy_from_legacy_root(args.source_root, model_root=args.model_root, replace=args.replace)
        return 0
    if args.command == "doctor":
        return doctor.run(args.model_root)
    if args.command == "engine-check":
        for module_name in import_all():
            print(f"ok: {module_name}")
        return 0
    if args.command == "make-sample":
        from .devdata import write_sample_sqlite

        path = write_sample_sqlite(args.output, frames=args.frames)
        print(path)
        return 0
    if args.command == "smoke":
        return run_polygon_smoke(args.work_dir, force=args.force)
    return 2
