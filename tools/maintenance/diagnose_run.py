#!/usr/bin/env python3
"""Diagnose a completed run directory without rerunning inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.pipeline.run_audit import audit_run_dir, audit_to_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path, help="Optional file to write.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audit = audit_run_dir(args.run_dir)
    if args.format == "json":
        text = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    else:
        text = audit_to_markdown(audit)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

