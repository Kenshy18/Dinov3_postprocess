"""Helpers for calling legacy-style argparse entrypoints in-process."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence


def run_argparse_main(func: Callable[[], object], argv: Sequence[str]) -> int:
    previous_argv = sys.argv[:]
    sys.argv = [getattr(func, "__name__", "stage"), *[str(arg) for arg in argv]]
    try:
        func()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = previous_argv
    return 0
