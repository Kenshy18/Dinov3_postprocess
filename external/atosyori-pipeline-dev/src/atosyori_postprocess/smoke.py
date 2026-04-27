"""Runnable smoke checks for development."""

from __future__ import annotations

import shutil
from pathlib import Path

from .devdata import write_sample_sqlite
from .stages import polygon_keyframes


def run_polygon_smoke(work_dir: Path, *, force: bool = False) -> int:
    if work_dir.exists() and force:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    input_sqlite = write_sample_sqlite(work_dir / "input.sqlite", frames=8)
    output_dir = work_dir / "polygon"

    args = [
        "--input-sqlite",
        str(input_sqlite),
        "--output-dir",
        str(output_dir),
        "--target-ratio",
        "0.5",
        "--anchors-per-contour",
        "8",
        "--no-adaptive-anchor-counts",
        "--num-workers",
        "1",
        "--evaluate-exact",
        "--write-pred-sqlite",
    ]
    code = polygon_keyframes.run(args)
    if code != 0:
        return code

    expected = output_dir / "summary.json"
    if not expected.is_file():
        print(f"missing smoke summary: {expected}")
        return 1
    print(f"smoke ok: {expected}")
    return 0
