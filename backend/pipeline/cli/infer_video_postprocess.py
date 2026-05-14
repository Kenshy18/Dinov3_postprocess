#!/usr/bin/env python3
"""User-facing video inference + postprocess entrypoint.

This is a thin wrapper over run_integrated_pipeline.py with a smaller CLI for
the common production path:

video -> detector JSONL/classification -> Atosyori SQLite -> optional overlay.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from backend.pipeline.pipeline_defaults import DINO_DEFAULT_BATCH_SIZE, DINO_DEFAULT_WARMUP_FRAMES


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
PIPELINE_SCRIPT = SCRIPT_DIR.parent / "run_integrated_pipeline.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run detector video inference and Atosyori postprocess")
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output" / "runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--detector", choices=("dinov3", "eva02"), default="dinov3")
    parser.add_argument("--classifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ellipse-only", action="store_true", help="Use ellipse-only class policy")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments forwarded to run_integrated_pipeline.py after '--'.",
    )
    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--input",
        str(args.input),
        "--output-root",
        str(args.output_root),
        "--detector",
        str(args.detector),
        "--classifier" if args.classifier else "--no-classifier",
        "--postprocess",
        "--render-overlays" if args.overlay else "--no-render-overlays",
    ]
    if args.detector == "eva02":
        if args.warmup_frames is not None:
            command.extend(["--eva02-warmup-frames", str(args.warmup_frames)])
        if args.batch_size is not None:
            command.extend(["--eva02-batch-size", str(args.batch_size)])
    else:
        command.extend(
            ["--warmup-frames", str(DINO_DEFAULT_WARMUP_FRAMES if args.warmup_frames is None else args.warmup_frames)]
        )
        command.extend(["--batch-size", str(DINO_DEFAULT_BATCH_SIZE if args.batch_size is None else args.batch_size)])
    if args.run_name:
        command.extend(["--run-name", args.run_name])
    if args.recursive:
        command.append("--recursive")
    if args.force:
        command.append("--force")
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.ellipse_only:
        command.extend(["--class-policy-json", str(ROOT / "configs" / "class_policy_ellipse_only.json")])
    if args.extra_args:
        forwarded = args.extra_args[1:] if args.extra_args[0] == "--" else args.extra_args
        command.extend(forwarded)
    return command


def main() -> int:
    args = build_parser().parse_args()
    command = build_command(args)
    print("[cmd] " + " ".join(command), flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
