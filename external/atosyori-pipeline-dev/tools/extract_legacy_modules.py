#!/usr/bin/env python3
"""Extract importable engine modules from legacy/run_standalone.py.

This is a mechanical developer tool. It does not delete or rewrite the legacy
engine; it mirrors named sections into src/atosyori_postprocess/engine so the
code can be inspected and gradually refactored by subsystem.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "src" / "atosyori_postprocess" / "legacy" / "run_standalone.py"
ENGINE_DIR = ROOT / "src" / "atosyori_postprocess" / "engine"

REGISTER_PRELUDE = '''\
from __future__ import annotations

import sys
import types
from pathlib import Path

SELF_PATH = Path(__file__).resolve()
ROOT = SELF_PATH.parent


def _register_inline_module(module_name: str, export_map: dict[str, str]) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = str(SELF_PATH)
    for public_name, global_name in export_map.items():
        value = globals()[global_name]
        setattr(module, public_name, value)
        setattr(module, global_name, value)
    sys.modules[module_name] = module
    return module

'''

SIMPLE_PRELUDE = '''\
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

'''

ELLIPSE_PRELUDE = SIMPLE_PRELUDE + '''\
import subprocess

try:
    import orjson
except ModuleNotFoundError:
    orjson = None

'''

RENDER_PRELUDE = SIMPLE_PRELUDE + "import standalone_runtime_fst\n\n"

SECTIONS = {
    "standalone_runtime_fst.py": ("Inlined from: standalone_runtime_fst.py", REGISTER_PRELUDE),
    "standalone_runtime_k2v5.py": ("Inlined from: standalone_runtime_k2v5.py", REGISTER_PRELUDE),
    "ellipse_inference.py": ("Inlined from: final_standalone_t5000_k1_exact_k2_v5.py", ELLIPSE_PRELUDE),
    "optimize_keyframes_standalone.py": (
        "Inlined from: keyframe_opt/optimize_keyframes_standalone.py",
        REGISTER_PRELUDE,
    ),
    "optimize_keyframes_dense_recall_standalone.py": (
        "Inlined from: keyframe_opt/optimize_keyframes_dense_recall_standalone.py",
        REGISTER_PRELUDE,
    ),
    "optimize_keyframes_trackk_dense_recall_standalone.py": (
        "Inlined from: keyframe_opt/optimize_keyframes_trackk_dense_recall_standalone.py",
        REGISTER_PRELUDE,
    ),
    "evaluate_keyframes_exact.py": ("Inlined from: keyframe_opt/evaluate_keyframes_exact.py", SIMPLE_PRELUDE),
    "fill_trackk_union_gaps.py": ("Inlined from: keyframe_opt/fill_trackk_union_gaps.py", SIMPLE_PRELUDE),
    "union_json_to_pred_sqlite.py": ("Inlined from: keyframe_opt/union_json_to_pred_sqlite.py", SIMPLE_PRELUDE),
    "render_overlay.py": ("Inlined from: render_k1_exact_k2_v5_overlay_video.py", RENDER_PRELUDE),
}


def marker_line_indices(lines: list[str]) -> list[int]:
    return [idx for idx, line in enumerate(lines) if "Inlined from:" in line or "Full pipeline orchestration" in line]


def section_body(lines: list[str], marker: str) -> str:
    starts = [idx for idx, line in enumerate(lines) if marker in line]
    if not starts:
        raise RuntimeError(f"marker not found: {marker}")
    marker_idx = starts[0]
    start = marker_idx + 2
    markers = marker_line_indices(lines)
    next_markers = [idx for idx in markers if idx > marker_idx]
    end = min(next_markers) - 1 if next_markers else len(lines)
    return "\n".join(lines[start:end]).strip() + "\n"


def polygon_v22_body(lines: list[str]) -> str:
    begin = "# --- BEGIN READABLE EMBEDDED POLYGON V22 SOURCE ---"
    end = "# --- END READABLE EMBEDDED POLYGON V22 SOURCE ---"
    begin_idx = next(idx for idx, line in enumerate(lines) if line.strip() == begin)
    end_idx = next(idx for idx in range(begin_idx + 1, len(lines)) if lines[idx].strip() == end)
    return textwrap.dedent("\n".join(lines[begin_idx + 1 : end_idx])).strip() + "\n"


def write_module(path: Path, prelude: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prelude + "\n" + body, encoding="utf-8")
    print(path.relative_to(ROOT))


def main() -> int:
    lines = LEGACY.read_text(encoding="utf-8").splitlines()
    for filename, (marker, prelude) in SECTIONS.items():
        write_module(ENGINE_DIR / filename, prelude, section_body(lines, marker))
    write_module(ENGINE_DIR / "polygon_v22.py", SIMPLE_PRELUDE, polygon_v22_body(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
