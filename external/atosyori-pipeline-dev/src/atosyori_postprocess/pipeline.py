"""High-level pipeline command builder."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from .legacy import legacy_command, run_legacy
from .settings import resolve_models


def choose_device(value: str) -> str:
    if value != "auto":
        return value
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_command(args: Namespace) -> list[str]:
    models = resolve_models(args.model_root)
    k2_dir = args.k2_run_dir or models.k2_dir
    polygon_dir = args.polygon_point_predictor_model_dir or models.polygon_point_predictor_dir

    engine_args: list[str] = [
        "--output-dir",
        str(args.output_dir),
        "--intervals",
        str(args.intervals),
        "--k2-run-dir",
        str(k2_dir),
        "--k2-device",
        choose_device(args.k2_device),
        "--polygon-point-predictor-model-dir",
        str(polygon_dir),
        "--polygon-predictor-device",
        choose_device(args.polygon_predictor_device),
    ]

    if args.input_sqlite is not None:
        engine_args.extend(["--input-sqlite", str(args.input_sqlite)])
    else:
        engine_args.extend(["--input-jsonl", str(args.input_jsonl)])
    if args.input_video is not None:
        engine_args.extend(["--input-video", str(args.input_video)])
    if args.default_shape_mode is not None:
        engine_args.extend(["--default-shape-mode", str(args.default_shape_mode)])
    if args.class_policy_json is not None:
        engine_args.extend(["--class-policy-json", str(args.class_policy_json)])
    if args.render_overlays is not None:
        engine_args.append("--render-overlays" if args.render_overlays else "--no-render-overlays")
    if args.force:
        engine_args.append("--force")
    if args.engine_args:
        extra = list(args.engine_args)
        if extra and extra[0] == "--":
            extra = extra[1:]
        engine_args.extend(extra)

    return legacy_command(engine_args, python=sys.executable)


def run(args: Namespace) -> int:
    command = build_command(args)
    return run_legacy(command[2:], python=command[0])
