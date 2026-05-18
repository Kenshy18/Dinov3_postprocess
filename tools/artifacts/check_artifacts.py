#!/usr/bin/env python3
"""Check local runtime artifact placement."""

from __future__ import annotations

import argparse
import json

try:
    from .runtime_artifacts import (
        ALL_RUNTIME_ARTIFACTS,
        DETECTRON2_EXTENSION_ROOT,
        REQUIRED_ARTIFACTS,
        ROOT,
        RTDETR_DEFAULT_CONFIG,
        RTDETR_RUNTIME_SCRIPT,
    )
except ImportError:  # pragma: no cover - direct script execution
    from runtime_artifacts import (
        ALL_RUNTIME_ARTIFACTS,
        DETECTRON2_EXTENSION_ROOT,
        REQUIRED_ARTIFACTS,
        ROOT,
        RTDETR_DEFAULT_CONFIG,
        RTDETR_RUNTIME_SCRIPT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check integrated runtime artifacts")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--allow-missing", action="store_true", help="Return 0 even if required files are missing")
    parser.add_argument("--require-trt", action="store_true", help="Require locally built TensorRT engines too")
    args = parser.parse_args()

    required = ALL_RUNTIME_ARTIFACTS if args.require_trt else REQUIRED_ARTIFACTS
    rows = []
    for artifact in ALL_RUNTIME_ARTIFACTS:
        rows.append({"name": artifact.name, "path": str(artifact.path), "exists": artifact.path.is_file()})

    rows.append(
        {
            "name": "Detectron2 CUDA extension",
            "path": str(DETECTRON2_EXTENSION_ROOT / "_C*.so"),
            "exists": any(DETECTRON2_EXTENSION_ROOT.glob("_C*.so")),
        }
    )
    source_rows = [
        {
            "name": "RT-DETR Head/Face runtime script",
            "path": str(RTDETR_RUNTIME_SCRIPT),
            "exists": RTDETR_RUNTIME_SCRIPT.is_file(),
        },
        {
            "name": "RT-DETR Head/Face config",
            "path": str(RTDETR_DEFAULT_CONFIG),
            "exists": RTDETR_DEFAULT_CONFIG.is_file(),
        },
    ]
    rows.extend(source_rows)

    if args.json:
        print(json.dumps({"root": str(ROOT), "artifacts": rows}, ensure_ascii=False, indent=2))
    else:
        print(f"[ARTIFACTS] root: {ROOT}")
        for row in rows:
            status = "ok" if row["exists"] else "missing"
            print(f"[{status}] {row['name']}: {row['path']}")

    required_paths = {str(artifact.path) for artifact in required}
    required_paths.update(str(row["path"]) for row in source_rows)
    missing_required = [row for row in rows if row["path"] in required_paths and not row["exists"]]
    if missing_required and not args.allow_missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
