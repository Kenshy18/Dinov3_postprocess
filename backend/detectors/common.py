"""Shared helpers for backend-facing detector adapters."""

from __future__ import annotations

from pathlib import Path


def maybe_add(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def require_script(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
