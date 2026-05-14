#!/usr/bin/env python3
"""Inventory cleanup candidates without deleting anything."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
WRAPPER_MARKERS = (
    "Compatibility wrapper",
    "Compatibility entrypoint",
    "compatibility wrapper",
    "compatibility entrypoint",
    "互換",
)
WRAPPER_ROOTS = ("scripts", "tools", "UI", "inference")
GENERATED_ROOTS = ("output", ".runtime", ".serena", ".pytest_cache", ".ruff_cache")
LARGE_SOURCE_ROOTS = ("backend", "external", "training")


@dataclass(frozen=True)
class CleanupCandidate:
    path: str
    category: str
    reason: str
    confidence: str
    action: str
    size_bytes: int = 0
    lines: int | None = None


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def text_file_lines(path: Path) -> int | None:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return None


def iter_files(root: Path, names: Iterable[str]) -> Iterable[Path]:
    for name in names:
        path = root / name
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from (p for p in path.rglob("*") if p.is_file())


def is_wrapper(path: Path) -> bool:
    if path.suffix not in {".py", ".sh", ".ps1"}:
        return False
    lines = text_file_lines(path)
    if lines is None or lines > 80:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    return any(marker in text for marker in WRAPPER_MARKERS) or "runpy.run_module" in text or "exec " in text


def inventory(root: Path = ROOT) -> list[CleanupCandidate]:
    root = root.resolve()
    candidates: list[CleanupCandidate] = []

    for path in iter_files(root, WRAPPER_ROOTS):
        if is_wrapper(path):
            candidates.append(
                CleanupCandidate(
                    path=rel(path, root),
                    category="compatibility_wrapper",
                    reason="Thin legacy entrypoint; keep until user shortcuts and docs no longer need it.",
                    confidence="review_required",
                    action="confirm_before_delete",
                    size_bytes=path.stat().st_size,
                    lines=text_file_lines(path),
                )
            )

    for name in GENERATED_ROOTS:
        path = root / name
        if not path.exists():
            continue
        candidates.append(
            CleanupCandidate(
                path=rel(path, root),
                category="local_generated_state",
                reason="Generated or machine-local state; use tools/clean_generated.py for controlled cleanup.",
                confidence="safe_to_clean_with_tool",
                action="clean_with_existing_tool",
                size_bytes=sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size,
                lines=None,
            )
        )

    for path in iter_files(root, LARGE_SOURCE_ROOTS):
        if path.suffix != ".py":
            continue
        lines = text_file_lines(path)
        if lines is not None and lines >= 1000:
            candidates.append(
                CleanupCandidate(
                    path=rel(path, root),
                    category="large_source_file",
                    reason="Large implementation file; candidate for extraction or module split, not direct deletion.",
                    confidence="manual_refactor_only",
                    action="do_not_delete_without_owner_review",
                    size_bytes=path.stat().st_size,
                    lines=lines,
                )
            )

    return sorted(candidates, key=lambda item: (item.category, item.path))


def to_markdown(candidates: list[CleanupCandidate]) -> str:
    lines = [
        "# Cleanup Candidate Inventory",
        "",
        "This is an inventory only. It does not approve deletion.",
        "",
        "| Category | Path | Lines | Confidence | Action | Reason |",
        "|---|---:|---:|---|---|---|",
    ]
    for item in candidates:
        lines.append(
            "| "
            + " | ".join(
                [
                    item.category,
                    f"`{item.path}`",
                    "" if item.lines is None else str(item.lines),
                    item.confidence,
                    item.action,
                    item.reason,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    candidates = inventory(args.root)
    if args.format == "json":
        print(json.dumps([asdict(item) for item in candidates], ensure_ascii=False, indent=2))
    else:
        print(to_markdown(candidates), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
