"""Minimal wandb stub for inference-only bundled runtime.

The bundled EVA02/Detectron2 code imports wandb from training utilities even
when only inference is used. This stub keeps those imports harmless without
requiring the optional wandb package.
"""

from __future__ import annotations


class _Run:
    name = ""


run = _Run()


def init(*args, **kwargs):
    return run


def log(*args, **kwargs) -> None:
    return None


def finish(*args, **kwargs) -> None:
    return None
