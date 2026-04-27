"""Shared helpers for stage wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..pipeline import choose_device
from ..settings import resolve_models


def has_option(args: Sequence[str], option: str) -> bool:
    prefix = option + "="
    return any(str(arg) == option or str(arg).startswith(prefix) for arg in args)


def append_if_missing(args: list[str], option: str, value: str | Path) -> None:
    if not has_option(args, option):
        args.extend([option, str(value)])


def with_ellipse_defaults(args: Sequence[str], model_root: Path | None = None) -> list[str]:
    output = [str(arg) for arg in args]
    models = resolve_models(model_root)
    append_if_missing(output, "--k2-run-dir", models.k2_dir)
    append_if_missing(output, "--k2-device", choose_device("auto"))
    return output


def with_polygon_defaults(args: Sequence[str], model_root: Path | None = None) -> list[str]:
    output = [str(arg) for arg in args]
    models = resolve_models(model_root)
    append_if_missing(output, "--point-predictor-model-dir", models.polygon_point_predictor_dir)
    append_if_missing(output, "--predictor-device", choose_device("auto"))
    return output
