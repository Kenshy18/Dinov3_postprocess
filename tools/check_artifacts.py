#!/usr/bin/env python3
"""Check local runtime artifact placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "detector checkpoint": ROOT / "checkpoints" / "detector" / "model_final.pth",
    "DINOv3 pretrained weights": ROOT
    / "checkpoints"
    / "dinov3"
    / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "ROI classifier checkpoint": ROOT / "checkpoints" / "classifier" / "best.pt",
    "EVA02 detector checkpoint": ROOT / "checkpoints" / "eva02" / "detector" / "model_final.pth",
    "EVA02 ROI classifier checkpoint": ROOT / "checkpoints" / "eva02" / "classifier" / "best.pt",
    "TensorRT backbone engine": ROOT
    / "checkpoints"
    / "trt"
    / "dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine",
    "postprocess K2 checkpoint": ROOT / "checkpoints" / "postprocess" / "k2_v5" / "best_exact.pt",
    "postprocess polygon checkpoint": ROOT
    / "checkpoints"
    / "postprocess"
    / "polygon_point_predictor"
    / "best.pt",
    "postprocess polygon feature stats": ROOT
    / "checkpoints"
    / "postprocess"
    / "polygon_point_predictor"
    / "feature_stats.npz",
}

OPTIONAL_GENERATED = {
    "Detectron2 CUDA extension": ROOT / "eva02" / "eva02_det" / "detectron2",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check integrated runtime artifacts")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--allow-missing", action="store_true", help="Return 0 even if required files are missing")
    args = parser.parse_args()

    rows = []
    for name, path in REQUIRED.items():
        rows.append({"name": name, "path": str(path), "exists": path.is_file()})

    extension_root = OPTIONAL_GENERATED["Detectron2 CUDA extension"]
    rows.append(
        {
            "name": "Detectron2 CUDA extension",
            "path": str(extension_root / "_C*.so"),
            "exists": any(extension_root.glob("_C*.so")),
        }
    )

    if args.json:
        print(json.dumps({"root": str(ROOT), "artifacts": rows}, ensure_ascii=False, indent=2))
    else:
        print(f"[ARTIFACTS] root: {ROOT}")
        for row in rows:
            status = "ok" if row["exists"] else "missing"
            print(f"[{status}] {row['name']}: {row['path']}")

    missing_required = [row for row in rows[: len(REQUIRED)] if not row["exists"]]
    if missing_required and not args.allow_missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
