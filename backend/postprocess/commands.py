"""Command and environment builders for the Atosyori postprocess engine."""

from __future__ import annotations

import os
from pathlib import Path


def require_atosyori_repo(atosyori_repo: Path) -> Path:
    src = atosyori_repo / "src" / "atosyori_postprocess"
    if not src.is_dir():
        raise FileNotFoundError(src)
    return atosyori_repo


def build_env(atosyori_repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    src = require_atosyori_repo(atosyori_repo) / "src"
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src) if not current else str(src) + os.pathsep + current
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def build_run_command(
    *,
    python: Path,
    atosyori_repo: Path,
    output_dir: Path,
    model_root: Path,
    intervals: int,
    default_shape_mode: str,
    k2_device: str,
    polygon_predictor_device: str,
    render_overlays: bool,
    force: bool,
    input_jsonl: Path | None = None,
    input_sqlite: Path | None = None,
    input_video: Path | None = None,
    class_policy_json: Path | None = None,
    engine_args: list[str] | None = None,
) -> list[str]:
    require_atosyori_repo(atosyori_repo)
    if input_jsonl is None and input_sqlite is None:
        raise ValueError("input_jsonl or input_sqlite is required")
    if input_jsonl is not None and input_sqlite is not None:
        raise ValueError("input_jsonl and input_sqlite are mutually exclusive")

    command = [
        str(python),
        "-m",
        "atosyori_postprocess",
        "run",
    ]
    if input_jsonl is not None:
        command.extend(["--input-jsonl", str(input_jsonl)])
    else:
        command.extend(["--input-sqlite", str(input_sqlite)])
    if input_video is not None:
        command.extend(["--input-video", str(input_video)])
    command.extend(
        [
            "--output-dir",
            str(output_dir),
            "--model-root",
            str(model_root),
            "--intervals",
            str(intervals),
            "--default-shape-mode",
            str(default_shape_mode),
            "--k2-device",
            str(k2_device),
            "--polygon-predictor-device",
            str(polygon_predictor_device),
            "--render-overlays" if render_overlays else "--no-render-overlays",
        ]
    )
    if class_policy_json is not None:
        command.extend(["--class-policy-json", str(class_policy_json)])
    if force:
        command.append("--force")
    if engine_args:
        command.append("--")
        command.extend(engine_args)
    return command
