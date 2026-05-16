#!/usr/bin/env python3
"""Clean postprocess-only entrypoint for existing detector JSONL or tracked SQLite."""

from __future__ import annotations

import argparse
import os
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
from backend.pipeline.pipeline_defaults import DEFAULT_MODEL_ROOT, LOCAL_ATOSYORI_REPO
from backend.pipeline.pipeline_outputs import collect_postprocess_outputs
from backend.postprocess.commands import build_env as build_atosyori_env
from backend.postprocess.commands import build_run_command as build_atosyori_run_command


def default_atosyori_repo() -> Path:
    return Path(os.environ.get("ATOSYORI_REPO", str(LOCAL_ATOSYORI_REPO)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run only the Atosyori postprocess pipeline on detector JSONL or tracked SQLite."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-jsonl", type=Path)
    input_group.add_argument("--input-sqlite", type=Path)
    parser.add_argument("--input-video", type=Path, default=None)

    parser.add_argument("--output-root", type=Path, default=ROOT / "output" / "postprocess_runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--atosyori-repo", type=Path, default=default_atosyori_repo())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overlay-encoder", choices=("cpu", "nvenc"), default="cpu")
    add_policy_args(parser)

    parser.add_argument("--raw-cut-detect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--short-track-max-frames", type=int, default=10)
    parser.add_argument("--raw-det-score-min", type=float, default=0.35)
    parser.add_argument("--k2-device", default="auto")
    parser.add_argument("--polygon-predictor-device", default="auto")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Forward advanced engine arguments to atosyori-postprocess after '--'.",
    )
    return parser


def build_command(args: argparse.Namespace, output_dir: Path, policy_path: Path) -> list[str]:
    engine_args = [
        "--overlay-encoder",
        str(args.overlay_encoder),
        "--raw-remove-short-tracks-max-frames",
        str(args.short_track_max_frames),
        "--raw-cut-detect" if args.raw_cut_detect else "--no-raw-cut-detect",
        "--raw-det-score-min",
        str(args.raw_det_score_min),
        "--dense-recall-target",
        str(args.recall_target),
        "--polygon-recall-min",
        str(args.recall_target),
    ]
    engine_args.extend(strip_remainder(list(args.extra_args or [])))
    return build_atosyori_run_command(
        python=args.python,
        atosyori_repo=args.atosyori_repo,
        input_jsonl=args.input_jsonl,
        input_sqlite=args.input_sqlite,
        input_video=args.input_video,
        output_dir=output_dir,
        model_root=args.model_root,
        intervals=args.keyframe_interval,
        default_shape_mode=args.shape_mode,
        class_policy_json=policy_path,
        k2_device=args.k2_device,
        polygon_predictor_device=args.polygon_predictor_device,
        render_overlays=args.overlay,
        force=args.force,
        engine_args=engine_args,
    )


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.output_root = Path(args.output_root).expanduser().resolve()
    args.run_name = args.run_name or f"postprocess_{timestamp()}"
    args.output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else args.output_root / args.run_name / "postprocess"
    )
    args.model_root = Path(args.model_root).expanduser().resolve()
    args.atosyori_repo = Path(args.atosyori_repo).expanduser().resolve()
    args.python = Path(os.path.abspath(os.path.expanduser(str(args.python))))
    if args.input_jsonl is not None:
        args.input_jsonl = Path(args.input_jsonl).expanduser().resolve()
    if args.input_sqlite is not None:
        args.input_sqlite = Path(args.input_sqlite).expanduser().resolve()
    if args.input_video is not None:
        args.input_video = Path(args.input_video).expanduser().resolve()
    args.keyframe_interval = validate_interval(args.keyframe_interval)
    args.recall_target = validate_recall(args.recall_target)
    return args


def atosyori_env(args: argparse.Namespace) -> dict[str, str]:
    return build_atosyori_env(args.atosyori_repo)


def main() -> int:
    args = normalize_args(build_parser().parse_args())
    run_dir = args.output_dir.parent
    policy_path = resolve_policy_path(args, run_dir / "config")
    command = build_command(args, args.output_dir, policy_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / "postprocess.log"
    request_summary = {
        "run_dir": str(run_dir),
        "output_dir": str(args.output_dir),
        "policy_json": str(policy_path),
        "log": str(log_path),
        "command": command,
        "command_text": command_to_text(command),
        "dry_run": bool(args.dry_run),
    }
    write_json(run_dir / "postprocess_request.json", request_summary)

    if args.dry_run:
        print("[dry-run] " + command_to_text(command), flush=True)
        print(f"[request] {run_dir / 'postprocess_request.json'}", flush=True)
        return 0

    returncode, elapsed = run_logged(
        command,
        log_path=log_path,
        cwd=args.atosyori_repo,
        env=atosyori_env(args),
    )
    request_summary.update({"returncode": returncode, "elapsed_seconds": elapsed})
    summary_path = args.output_dir / "summary.json"
    if returncode == 0 and summary_path.is_file():
        request_summary["postprocess_summary"] = str(summary_path)
        if args.input_video is not None:
            request_summary["outputs"] = collect_postprocess_outputs(run_dir, args.output_dir, args.input_video)
    write_json(run_dir / "postprocess_request.json", request_summary)

    if returncode != 0:
        print(f"[error] postprocess failed; see log: {log_path}", file=sys.stderr, flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
