"""Production-patched polygon optimizer runtime.

The readable ``polygon_v22.py`` module is the extracted solver source. The
validated standalone also applies a set of production patches around that
solver for degenerate real-world contours, cached exact evaluation, compact
JSON output, and stable multiprocessing. Until those patches are fully moved
into explicit modules, this runtime exposes the validated patched module.
"""

from __future__ import annotations

from ..legacy import run_stage

STAGE = "__onefile_polygon_optimize"


def run(args: list[str]) -> int:
    return run_stage(STAGE, args)
