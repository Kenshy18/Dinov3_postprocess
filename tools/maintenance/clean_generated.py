#!/usr/bin/env python3
"""Clean generated local files with a dry-run default."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKIP_DIR_NAMES = {".git", ".venv_integrated"}
OUTPUT_KEEP_NAMES = {".gitkeep", "README.md"}


@dataclass(frozen=True)
class CleanupTarget:
    path: Path
    reason: str


def _is_under_skipped_dir(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in SKIP_DIR_NAMES for part in rel_parts)


def collect_targets(
    root: Path,
    *,
    include_caches: bool,
    include_output: bool,
    include_serena: bool,
) -> list[CleanupTarget]:
    root = root.resolve()
    targets: list[CleanupTarget] = []

    if include_caches:
        for path in root.rglob("__pycache__"):
            if path.is_dir() and not _is_under_skipped_dir(path, root):
                targets.append(CleanupTarget(path, "python bytecode cache"))
        for name in (".pytest_cache", ".ruff_cache"):
            path = root / name
            if path.exists():
                targets.append(CleanupTarget(path, "tool cache"))

    if include_output:
        output_dir = root / "output"
        if output_dir.is_dir():
            for child in sorted(output_dir.iterdir()):
                if child.name not in OUTPUT_KEEP_NAMES:
                    targets.append(CleanupTarget(child, "generated output"))

    if include_serena:
        path = root / ".serena"
        if path.exists():
            targets.append(CleanupTarget(path, "local Serena metadata"))

    return sorted(targets, key=lambda item: str(item.path))


def remove_target(target: CleanupTarget) -> None:
    path = target.path
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean generated local project files.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--caches", action="store_true", help="Clean __pycache__, pytest, and ruff caches.")
    parser.add_argument("--output", action="store_true", help="Clean output/ while preserving .gitkeep and README.md.")
    parser.add_argument("--serena", action="store_true", help="Clean .serena local metadata.")
    parser.add_argument("--all-local", action="store_true", help="Clean caches, output, and .serena.")
    parser.add_argument("--apply", action="store_true", help="Actually delete targets. Default is dry-run.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    include_caches = bool(args.caches or args.all_local)
    include_output = bool(args.output or args.all_local)
    include_serena = bool(args.serena or args.all_local)
    if not (include_caches or include_output or include_serena):
        print("[clean] no target group selected; use --caches, --output, --serena, or --all-local")
        return 2

    targets = collect_targets(
        Path(args.root),
        include_caches=include_caches,
        include_output=include_output,
        include_serena=include_serena,
    )
    action = "delete" if args.apply else "would-delete"
    for target in targets:
        print(f"[clean] {action}: {target.path} ({target.reason})")
        if args.apply:
            remove_target(target)
    print(f"[clean] targets={len(targets)} apply={bool(args.apply)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
