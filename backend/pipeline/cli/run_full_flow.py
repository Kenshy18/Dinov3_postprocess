#!/usr/bin/env python3
"""Clean end-to-end entrypoint: video -> AI JSONL -> optional postprocess."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .flow_cli_common import (
    ROOT,
    add_policy_args,
    command_to_text,
    resolve_policy_path,
    run_logged,
    strip_remainder,
    timestamp,
    validate_interval,
    validate_recall,
    write_json,
)


PIPELINE_SCRIPT = Path(__file__).resolve().parents[1] / "run_integrated_pipeline.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run detector inference and, by default, the full Atosyori postprocess pipeline."
    )
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output" / "runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--detector", choices=("dinov3", "eva02", "codino"), default="dinov3")
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overlay-encoder", choices=("cpu", "nvenc"), default="cpu")

    add_policy_args(parser)

    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=None)
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--raw-cut-detect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--short-track-max-frames", type=int, default=10)
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Forward advanced arguments to run_integrated_pipeline.py after '--'.",
    )
    return parser


def build_command(args: argparse.Namespace, run_dir: Path, policy_path: Path | None) -> list[str]:
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--input",
        str(args.input),
        "--output-root",
        str(args.output_root),
        "--run-name",
        str(args.run_name),
        "--detector",
        str(args.detector),
        "--postprocess" if args.postprocess else "--no-postprocess",
    ]

    if args.recursive:
        command.append("--recursive")
    if args.force:
        command.append("--force")
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])

    if args.detector == "eva02":
        if args.batch_size is not None:
            command.extend(["--eva02-batch-size", str(args.batch_size)])
        if args.warmup_frames is not None:
            command.extend(["--eva02-warmup-frames", str(args.warmup_frames)])
        if args.score_thresh is not None:
            command.extend(["--eva02-score-thresh", str(args.score_thresh)])
    elif args.detector == "codino":
        if args.batch_size is not None:
            command.extend(["--codino-batch-size", str(args.batch_size)])
        if args.warmup_frames is not None:
            command.extend(["--codino-warmup-frames", str(args.warmup_frames)])
        if args.score_thresh is not None:
            command.extend(["--codino-score-thresh", str(args.score_thresh)])
    else:
        if args.batch_size is not None:
            command.extend(["--batch-size", str(args.batch_size)])
        if args.warmup_frames is not None:
            command.extend(["--warmup-frames", str(args.warmup_frames)])
        if args.score_thresh is not None:
            command.extend(["--score-thresh", str(args.score_thresh)])

    if args.postprocess:
        command.extend(
            [
                "--intervals",
                str(args.keyframe_interval),
                "--default-shape-mode",
                str(args.shape_mode),
                "--render-overlays" if args.overlay else "--no-render-overlays",
                "--overlay-encoder",
                str(args.overlay_encoder),
                "--raw-cut-detect" if args.raw_cut_detect else "--no-raw-cut-detect",
                "--raw-remove-short-tracks-max-frames",
                str(args.short_track_max_frames),
            ]
        )
        if policy_path is not None:
            command.extend(["--class-policy-json", str(policy_path)])

    forwarded = strip_remainder(list(args.extra_args or []))
    if forwarded:
        command.extend(forwarded)

    if args.postprocess:
        command.extend(
            [
                "--postprocess-extra-args",
                "--dense-recall-target",
                str(args.recall_target),
                "--polygon-recall-min",
                str(args.recall_target),
            ]
        )
    return command


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.output_root = Path(args.output_root).expanduser().resolve()
    args.run_name = args.run_name or f"full_flow_{timestamp()}"
    args.keyframe_interval = validate_interval(args.keyframe_interval)
    args.recall_target = validate_recall(args.recall_target)
    return args


def main() -> int:
    args = normalize_args(build_parser().parse_args())
    run_dir = args.output_root / args.run_name
    policy_path = resolve_policy_path(args, run_dir / "config") if args.postprocess else None
    command = build_command(args, run_dir, policy_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / "full_flow.log"
    wrapper_summary = {
        "run_dir": str(run_dir),
        "policy_json": None if policy_path is None else str(policy_path),
        "log": str(log_path),
        "command": command,
        "command_text": command_to_text(command),
        "dry_run": bool(args.dry_run),
    }
    write_json(run_dir / "flow_request.json", wrapper_summary)

    if args.dry_run:
        print("[dry-run] " + command_to_text(command), flush=True)
        print(f"[request] {run_dir / 'flow_request.json'}", flush=True)
        return 0

    returncode, elapsed = run_logged(command, log_path=log_path, cwd=ROOT)
    wrapper_summary.update({"returncode": returncode, "elapsed_seconds": elapsed})
    write_json(run_dir / "flow_request.json", wrapper_summary)
    if returncode != 0:
        print(f"[error] full flow failed; see log: {log_path}", file=sys.stderr, flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
