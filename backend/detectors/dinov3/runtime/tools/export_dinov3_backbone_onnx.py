#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the DINOv3 Cascade backbone to ONNX for ORT/TensorRT EP tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# xFormers custom kernels are not exportable to ONNX. Force the SDPA path before
# importing the inference module, which imports DINOv3.
os.environ["DINOV3_USE_XFORMERS"] = "0"

import torch

RUNTIME_DIR = Path(__file__).resolve().parents[1]
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

import infer_images_singleclass as infer
from safe_fp16_backbone import apply_safe_fp16_islands

DEFAULT_DINOV3_WEIGHTS = (
    RUNTIME_DIR
    / "checkpoints"
    / "dinov3"
    / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)


class BackboneExportWrapper(torch.nn.Module):
    def __init__(
        self,
        backbone_net: torch.nn.Module,
        out_feature: str = "last_feat",
        use_autocast: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_net = backbone_net
        self.out_feature = out_feature
        self.use_autocast = use_autocast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_autocast and x.is_cuda:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = self.backbone_net(x)
        else:
            out = self.backbone_net(x)
        if isinstance(out, dict):
            return out[self.out_feature]
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out


class FullBackboneExportWrapper(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        output_names: tuple[str, ...],
        use_autocast: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_names = output_names
        self.use_autocast = use_autocast

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.use_autocast and x.is_cuda:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = self.backbone(x)
        else:
            out = self.backbone(x)
        return tuple(out[name] for name in self.output_names)


def _dtype_from_args(fp16: bool, device: str) -> torch.dtype:
    if fp16 and device.startswith("cuda"):
        return torch.float16
    return torch.float32


def _provider_names() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


def _verify_with_ort(
    *,
    onnx_path: Path,
    wrapper: BackboneExportWrapper,
    dummy: torch.Tensor,
    dtype: torch.dtype,
    use_trt: bool,
) -> dict[str, object]:
    import onnxruntime as ort

    providers: list[object] = []
    available = set(ort.get_available_providers())
    if use_trt and "TensorrtExecutionProvider" in available:
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": "True",
                    "trt_engine_cache_path": str(onnx_path.parent / "ort_trt_cache"),
                    "trt_fp16_enable": "True" if dtype == torch.float16 else "False",
                },
            )
        )
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    with torch.inference_mode():
        torch_out = wrapper(dummy).detach()
    if dummy.is_cuda:
        torch.cuda.synchronize()

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    run_input = dummy.detach().cpu().numpy()

    t0 = time.perf_counter()
    ort_out_np = sess.run([output_name], {input_name: run_input})[0]
    ort_elapsed = time.perf_counter() - t0
    ort_out = torch.from_numpy(ort_out_np).to(torch_out.device)

    diff = (torch_out.float() - ort_out.float()).abs()
    denom = torch_out.float().abs().clamp_min(1e-6)
    rel = diff / denom
    return {
        "providers_requested": [p[0] if isinstance(p, tuple) else p for p in providers],
        "providers_in_use": sess.get_providers(),
        "input_shape": list(dummy.shape),
        "torch_output_shape": list(torch_out.shape),
        "ort_output_shape": list(ort_out.shape),
        "torch_output_dtype": str(torch_out.dtype),
        "ort_output_dtype": str(ort_out.dtype),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "max_rel_diff": float(rel.max().item()),
        "mean_rel_diff": float(rel.mean().item()),
        "ort_elapsed_sec": ort_elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export DINOv3 Cascade backbone to ONNX")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=str(infer.DEFAULT_CONFIG))
    parser.add_argument(
        "--backbone-weights",
        default=str(DEFAULT_DINOV3_WEIGHTS) if DEFAULT_DINOV3_WEIGHTS.is_file() else infer.unified_paths.DINOv3_WEIGHTS,
    )
    parser.add_argument(
        "--output",
        default=str(infer.REPO_ROOT / "output" / "onnx" / "dinov3_backbone_fp32_720x1280_dynamic.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument("--target-size", type=infer._parse_target_size, default=(720, 1280))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--nms-thresh", type=float, default=0.4)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size used for export")
    parser.add_argument("--fixed-batch", action="store_true", help="Disable dynamic batch axis")
    parser.add_argument(
        "--full-backbone",
        action="store_true",
        help="Export the full SimpleFeaturePyramid backbone with p2..p6 outputs instead of only DINOv3 net",
    )
    parser.add_argument("--feature-names", default="p2,p3,p4,p5,p6")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--amp-export",
        action="store_true",
        help="Trace the fp32 backbone under CUDA autocast instead of calling backbone.half().",
    )
    parser.add_argument(
        "--fp16-safe-islands",
        action="store_true",
        help="Use fp16 weights/inputs but keep LayerNorm, LayerScale, and residual adds in fp32 to avoid overflow.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify-ort", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verify-trt", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = infer._resolve_checkpoint(args.checkpoint)
    target_h, target_w = infer._unpack_size(args.target_size)
    dtype = torch.float32 if args.amp_export else _dtype_from_args(args.fp16, args.device)
    if args.amp_export and args.fp16_safe_islands:
        raise ValueError("--amp-export and --fp16-safe-islands are mutually exclusive")
    if args.fp16_safe_islands and dtype != torch.float16:
        raise ValueError("--fp16-safe-islands requires --fp16 on CUDA")
    if args.full_backbone and (args.fp16 or args.amp_export or args.fp16_safe_islands):
        raise ValueError("--full-backbone currently supports fp32 export only; use --no-fp16")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    print(f"[INFO] checkpoint: {checkpoint}")
    print(f"[INFO] target_size: {target_h}x{target_w}")
    print(f"[INFO] dtype: {dtype}")
    print(f"[INFO] available ORT providers: {_provider_names()}")

    model = infer._build_model(
        checkpoint=checkpoint,
        target_size=args.target_size,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        topk_per_image=args.topk,
        device=args.device,
        config_path=Path(args.config).expanduser().resolve(),
        backbone_weights=args.backbone_weights,
    )
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "net"):
        raise RuntimeError("model.backbone.net not found; cannot export backbone")

    feature_names = tuple(part.strip() for part in str(args.feature_names).split(",") if part.strip())
    if args.full_backbone:
        output_names = list(feature_names)
        wrapper = FullBackboneExportWrapper(
            model.backbone.eval(),
            output_names=feature_names,
            use_autocast=bool(args.amp_export),
        ).to(args.device).eval()
    else:
        output_names = ["last_feat"]
        backbone_net = model.backbone.net.eval()
        if dtype == torch.float16:
            if args.fp16_safe_islands:
                backbone_net = apply_safe_fp16_islands(backbone_net)
            else:
                backbone_net = backbone_net.half()
        wrapper = BackboneExportWrapper(backbone_net, use_autocast=bool(args.amp_export)).to(args.device).eval()

    batch = max(1, int(args.batch_size))
    dummy = torch.randn(batch, 3, target_h, target_w, device=args.device, dtype=dtype)
    dynamic_axes = None
    if not args.fixed_batch:
        dynamic_axes = {"input": {0: "batch"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    t0 = time.perf_counter()
    print(f"[ONNX] exporting -> {output_path}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    meta: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "output_onnx": str(output_path),
        "target_size": f"{target_h}x{target_w}",
        "batch_size": batch,
        "fixed_batch": bool(args.fixed_batch),
        "dtype": str(dtype),
        "amp_export": bool(args.amp_export),
        "fp16_safe_islands": bool(args.fp16_safe_islands),
        "full_backbone": bool(args.full_backbone),
        "feature_names": list(feature_names) if args.full_backbone else None,
        "opset": int(args.opset),
        "export_elapsed_sec": elapsed,
        "available_ort_providers": _provider_names(),
    }

    if args.verify_ort or args.verify_trt:
        meta["verification"] = _verify_with_ort(
            onnx_path=output_path,
            wrapper=wrapper,
            dummy=dummy,
            dtype=dtype,
            use_trt=bool(args.verify_trt),
        )

    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DONE] wrote {output_path}")
    print(f"[DONE] wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
