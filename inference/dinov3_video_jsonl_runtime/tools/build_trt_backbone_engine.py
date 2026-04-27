#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a native TensorRT engine for the exported DINOv3 backbone ONNX."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import tensorrt as trt

RUNTIME_DIR = Path(__file__).resolve().parents[1]
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from trt_backbone import build_engine_from_onnx


def _shape(value: str) -> tuple[int, int, int, int]:
    parts = [int(part) for part in value.lower().replace(",", "x").split("x") if part]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must be BxCxHxW")
    return tuple(parts)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build TensorRT DINOv3 backbone engine")
    parser.add_argument("--onnx", required=True, help="Input ONNX path")
    parser.add_argument("--engine", required=True, help="Output TensorRT engine path")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--min-shape", type=_shape, default=None)
    parser.add_argument("--opt-shape", type=_shape, default=None)
    parser.add_argument("--max-shape", type=_shape, default=None)
    parser.add_argument(
        "--force-layer-precision",
        action="store_true",
        help="Set each parsed layer precision/output type to the requested precision when TensorRT accepts it",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    t0 = time.perf_counter()
    engine_path = build_engine_from_onnx(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        min_shape=args.min_shape,
        opt_shape=args.opt_shape,
        max_shape=args.max_shape,
        workspace_bytes=int(args.workspace_gb * (1 << 30)),
        force_layer_precision=args.force_layer_precision,
    )
    elapsed = time.perf_counter() - t0

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    tensors = []
    if engine is not None:
        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            tensors.append(
                {
                    "name": name,
                    "mode": str(engine.get_tensor_mode(name)),
                    "dtype": str(engine.get_tensor_dtype(name)),
                    "shape": list(engine.get_tensor_shape(name)),
                }
            )
    meta = {
        "onnx": str(Path(args.onnx).expanduser().resolve()),
        "engine": str(engine_path),
        "precision": args.precision,
        "force_layer_precision": bool(args.force_layer_precision),
        "min_shape": list(args.min_shape) if args.min_shape else None,
        "opt_shape": list(args.opt_shape) if args.opt_shape else None,
        "max_shape": list(args.max_shape) if args.max_shape else None,
        "workspace_gb": args.workspace_gb,
        "build_elapsed_sec": elapsed,
        "tensors": tensors,
    }
    meta_path = engine_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
